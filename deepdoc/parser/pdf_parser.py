#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import os
import random
import re
import sys
import threading
from copy import deepcopy
from io import BytesIO
from timeit import default_timer as timer

from deepdoc.depend.nltk_manager import require_nltk_data
import numpy as np
import pdfplumber
import trio
import xgboost as xgb
from huggingface_hub import snapshot_download
from PIL import Image
from pypdf import PdfReader as pdf2_read

from ..depend import settings
from ..depend.file_utils import get_project_base_directory
from ..vision import OCR, LayoutRecognizer, Recognizer, TableStructureRecognizer
from ..depend.vision_llm_chunk import vision_llm_chunk as picture_vision_llm_chunk
from ..depend import rag_tokenizer
from ..depend.prompts import vision_llm_describe_prompt
from ..depend.settings import PARALLEL_DEVICES

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


def _multiprocess_page_worker_standalone(args):
    """独立的多进程worker函数，不依赖self实例变量（模块级别，可被pickle序列化）
    
    这个函数必须在模块级别定义，以便 multiprocessing 可以序列化它。
    
    Args:
        args: 元组，包含 (page_idx, img_bytes_data, chars_data, zoomin_val, device_id_val, mean_height_val)
    
    Returns:
        元组: (page_idx, boxes, mean_height_val, lefted_chars)
    """
    page_idx, img_bytes_data, chars_data, zoomin_val, device_id_val, mean_height_val = args
    
    # 在每个进程中重新创建 OCR 实例
    ocr_instance = OCR()
    
    # 反序列化图像：从PNG字节转换为PIL Image，再转为numpy数组
    img = Image.open(BytesIO(img_bytes_data))
    img_np = np.array(img)
    
    # 执行OCR检测
    bxs = ocr_instance.detect(img_np, device_id_val)
    
    # 处理检测结果
    if not bxs:
        return page_idx, [], mean_height_val, []
    
    bxs = [(line[0], line[1][0]) for line in bxs]
    bxs = Recognizer.sort_Y_firstly(
        [
            {"x0": b[0][0] / zoomin_val, "x1": b[1][0] / zoomin_val, "top": b[0][1] / zoomin_val, 
             "text": "", "txt": t, "bottom": b[-1][1] / zoomin_val, "chars": [], "page_number": page_idx + 1}
            for b, t in bxs
            if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
        ],
        mean_height_val / 3 if mean_height_val > 0 else 8,
    )
    
    # 合并chars到boxes
    lefted_chars = []
    for c in chars_data:
        ii = Recognizer.find_overlapped(c, bxs)
        if ii is None:
            lefted_chars.append(c)
            continue
        ch = c["bottom"] - c["top"]
        bh = bxs[ii]["bottom"] - bxs[ii]["top"]
        if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
            lefted_chars.append(c)
            continue
        bxs[ii]["chars"].append(c)
    
    # 处理chars，生成文本
    for b in bxs:
        if not b["chars"]:
            del b["chars"]
            continue
        m_ht = np.mean([c["height"] for c in b["chars"]])
        for c in Recognizer.sort_Y_firstly(b["chars"], m_ht):
            if c["text"] == " " and b["text"]:
                if re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", b["text"][-1]):
                    b["text"] += " "
            else:
                b["text"] += c["text"]
        del b["chars"]
    
    # 识别没有文本的boxes
    boxes_to_reg = []
    for b in bxs:
        if not b["text"]:
            left, right, top, bott = b["x0"] * zoomin_val, b["x1"] * zoomin_val, b["top"] * zoomin_val, b["bottom"] * zoomin_val
            b["box_image"] = ocr_instance.get_rotate_crop_image(
                img_np, 
                np.array([[left, top], [right, top], [right, bott], [left, bott]], dtype=np.float32)
            )
            boxes_to_reg.append(b)
        del b["txt"]
    
    # 批量识别
    if boxes_to_reg:
        texts = ocr_instance.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id_val)
        for i in range(len(boxes_to_reg)):
            boxes_to_reg[i]["text"] = texts[i]
            del boxes_to_reg[i]["box_image"]
    
    # 过滤空文本的boxes
    bxs = [b for b in bxs if b["text"]]
    
    # 计算mean_height
    if mean_height_val == 0 and bxs:
        mean_height_val = np.median([b["bottom"] - b["top"] for b in bxs])
    
    # 返回结果：page_idx, boxes, mean_height, lefted_chars
    return page_idx, bxs, mean_height_val, lefted_chars


@require_nltk_data(("tokenizers/punkt_tab", "punkt_tab"))
class RAGFlowPdfParser:
    def __init__(self, **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """

        self.ocr = OCR()
        self.parallel_limiter = None
        
        # Check if multi-session mode is enabled (for single GPU)
        gpu_sessions = int(os.environ.get("DEEPDOC_GPU_SESSIONS", "0"))
        use_multi_session = gpu_sessions > 0 and PARALLEL_DEVICES > 0
        
        # Determine number of parallel units (sessions or devices)
        if use_multi_session:
            num_parallel = gpu_sessions
        else:
            num_parallel = PARALLEL_DEVICES
        
        if num_parallel > 1:
            # 支持通过环境变量设置 CapacityLimiter 容量
            use_original = os.environ.get("DEEPDOC_USE_ORIGINAL", "0").lower() in ("1", "true", "yes")
            if use_original:
                limiter_capacity = 1  # 原始 deepdoc 行为：capacity=1
            else:
                limiter_capacity = int(os.environ.get("DEEPDOC_LIMITER_CAPACITY", "2"))  # 优化：默认 capacity=2
            self.parallel_limiter = [trio.CapacityLimiter(limiter_capacity) for _ in range(num_parallel)]
            mode_str = f"multi-session ({gpu_sessions} sessions)" if use_multi_session else f"multi-GPU ({PARALLEL_DEVICES} devices)"
            print(f"[CONCURRENCY DEBUG] Enabled parallel processing with {num_parallel} limiters ({mode_str}), capacity={limiter_capacity} each")
            logging.info(f"[CONCURRENCY] Enabled parallel processing with {num_parallel} limiters ({mode_str}), capacity={limiter_capacity} each")
        else:
            print(f"[CONCURRENCY DEBUG] Parallel processing DISABLED (num_parallel={num_parallel}, need > 1)")
            logging.info(f"[CONCURRENCY] Parallel processing DISABLED (num_parallel={num_parallel}, need > 1)")

        if hasattr(self, "model_speciess"):
            self.layouter = LayoutRecognizer("layout." + self.model_speciess)
        else:
            self.layouter = LayoutRecognizer("layout")
        self.tbl_det = TableStructureRecognizer()

        self.updown_cnt_mdl = xgb.Booster()
        if not settings.LIGHTEN:
            try:
                import torch.cuda

                if torch.cuda.is_available():
                    self.updown_cnt_mdl.set_param({"device": "cuda"})
            except Exception:
                logging.exception("RAGFlowPdfParser __init__")
        try:
            model_dir = os.path.join(get_project_base_directory(), "dict")
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))
        except Exception:
            model_dir = snapshot_download(repo_id="InfiniFlow/text_concat_xgb_v1.0", local_dir=os.path.join(get_project_base_directory(), "dict"), local_dir_use_symlinks=False)
            self.updown_cnt_mdl.load_model(os.path.join(model_dir, "updown_concat_xgb.model"))

        self.page_from = 0

    def __char_width(self, c):
        return (c["x1"] - c["x0"]) // max(len(c["text"]), 1)

    def __height(self, c):
        return c["bottom"] - c["top"]

    def _x_dis(self, a, b):
        return min(abs(a["x1"] - b["x0"]), abs(a["x0"] - b["x1"]), abs(a["x0"] + a["x1"] - b["x0"] - b["x1"]) / 2)

    def _y_dis(self, a, b):
        return (b["top"] + b["bottom"] - a["top"] - a["bottom"]) / 2

    def _match_proj(self, b):
        proj_patt = [
            r"第[零一二三四五六七八九十百]+章",
            r"第[零一二三四五六七八九十百]+[条节]",
            r"[零一二三四五六七八九十百]+[、是 　]",
            r"[\(（][零一二三四五六七八九十百]+[）\)]",
            r"[\(（][0-9]+[）\)]",
            r"[0-9]+(、|\.[　 ]|）|\.[^0-9./a-zA-Z_%><-]{4,})",
            r"[0-9]+\.[0-9.]+(、|\.[ 　])",
            r"[⚫•➢①② ]",
        ]
        return any([re.match(p, b["text"]) for p in proj_patt])

    def _updown_concat_features(self, up, down):
        w = max(self.__char_width(up), self.__char_width(down))
        h = max(self.__height(up), self.__height(down))
        y_dis = self._y_dis(up, down)
        LEN = 6
        tks_down = rag_tokenizer.tokenize(down["text"][:LEN]).split()
        tks_up = rag_tokenizer.tokenize(up["text"][-LEN:]).split()
        tks_all = up["text"][-LEN:].strip() + (" " if re.match(r"[a-zA-Z0-9]+", up["text"][-1] + down["text"][0]) else "") + down["text"][:LEN].strip()
        tks_all = rag_tokenizer.tokenize(tks_all).split()
        fea = [
            up.get("R", -1) == down.get("R", -1),
            y_dis / h,
            down["page_number"] - up["page_number"],
            up["layout_type"] == down["layout_type"],
            up["layout_type"] == "text",
            down["layout_type"] == "text",
            up["layout_type"] == "table",
            down["layout_type"] == "table",
            True if re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up["text"]) else False,
            True if re.search(r"[，：‘“、0-9（+-]$", up["text"]) else False,
            True if re.search(r"(^.?[/,?;:\]，。；：’”？！》】）-])", down["text"]) else False,
            True if re.match(r"[\(（][^\(\)（）]+[）\)]$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[，,][^。.]+$", up["text"]) else False,
            True if re.search(r"[\(（][^\)）]+$", up["text"]) and re.search(r"[\)）]", down["text"]) else False,
            self._match_proj(down),
            True if re.match(r"[A-Z]", down["text"]) else False,
            True if re.match(r"[A-Z]", up["text"][-1]) else False,
            True if re.match(r"[a-z0-9]", up["text"][-1]) else False,
            True if re.match(r"[0-9.%,-]+$", down["text"]) else False,
            up["text"].strip()[-2:] == down["text"].strip()[-2:] if len(up["text"].strip()) > 1 and len(down["text"].strip()) > 1 else False,
            up["x0"] > down["x1"],
            abs(self.__height(up) - self.__height(down)) / min(self.__height(up), self.__height(down)),
            self._x_dis(up, down) / max(w, 0.000001),
            (len(up["text"]) - len(down["text"])) / max(len(up["text"]), len(down["text"])),
            len(tks_all) - len(tks_up) - len(tks_down),
            len(tks_down) - len(tks_up),
            tks_down[-1] == tks_up[-1] if tks_down and tks_up else False,
            max(down["in_row"], up["in_row"]),
            abs(down["in_row"] - up["in_row"]),
            len(tks_down) == 1 and rag_tokenizer.tag(tks_down[0]).find("n") >= 0,
            len(tks_up) == 1 and rag_tokenizer.tag(tks_up[0]).find("n") >= 0,
        ]
        return fea

    @staticmethod
    def sort_X_by_page(arr, threshold):
        # sort using y1 first and then x1
        arr = sorted(arr, key=lambda r: (r["page_number"], r["x0"], r["top"]))
        for i in range(len(arr) - 1):
            for j in range(i, -1, -1):
                # restore the order using th
                if abs(arr[j + 1]["x0"] - arr[j]["x0"]) < threshold and arr[j + 1]["top"] < arr[j]["top"] and arr[j + 1]["page_number"] == arr[j]["page_number"]:
                    tmp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = tmp
        return arr

    def _has_color(self, o):
        if o.get("ncs", "") == "DeviceGray":
            if o["stroking_color"] and o["stroking_color"][0] == 1 and o["non_stroking_color"] and o["non_stroking_color"][0] == 1:
                if re.match(r"[a-zT_\[\]\(\)-]+", o.get("text", "")):
                    return False
        return True

    def _table_transformer_job(self, ZM):
        logging.debug("Table processing...")
        imgs, pos = [], []
        tbcnt = [0]
        MARGIN = 10
        self.tb_cpns = []
        assert len(self.page_layout) == len(self.page_images)
        for p, tbls in enumerate(self.page_layout):  # for page
            tbls = [f for f in tbls if f["type"] == "table"]
            tbcnt.append(len(tbls))
            if not tbls:
                continue
            for tb in tbls:  # for table
                left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, tb["x1"] + MARGIN, tb["bottom"] + MARGIN
                left *= ZM
                top *= ZM
                right *= ZM
                bott *= ZM
                pos.append((left, top))
                imgs.append(self.page_images[p].crop((left, top, right, bott)))

        assert len(self.page_images) == len(tbcnt) - 1
        if not imgs:
            return
        recos = self.tbl_det(imgs)
        tbcnt = np.cumsum(tbcnt)
        for i in range(len(tbcnt) - 1):  # for page
            pg = []
            for j, tb_items in enumerate(recos[tbcnt[i] : tbcnt[i + 1]]):  # for table
                poss = pos[tbcnt[i] : tbcnt[i + 1]]
                for it in tb_items:  # for table components
                    it["x0"] = it["x0"] + poss[j][0]
                    it["x1"] = it["x1"] + poss[j][0]
                    it["top"] = it["top"] + poss[j][1]
                    it["bottom"] = it["bottom"] + poss[j][1]
                    for n in ["x0", "x1", "top", "bottom"]:
                        it[n] /= ZM
                    it["top"] += self.page_cum_height[i]
                    it["bottom"] += self.page_cum_height[i]
                    it["pn"] = i
                    it["layoutno"] = j
                    pg.append(it)
            self.tb_cpns.extend(pg)

        def gather(kwd, fzy=10, ption=0.6):
            eles = Recognizer.sort_Y_firstly([r for r in self.tb_cpns if re.match(kwd, r["label"])], fzy)
            eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
            return Recognizer.sort_Y_firstly(eles, 0)

        # add R,H,C,SP tag to boxes within table layout
        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted([r for r in self.tb_cpns if re.match(r"table column$", r["label"])], key=lambda x: (x["pn"], x["layoutno"], x["x0"]))
        clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)
        for b in self.boxes:
            if b.get("layout_type", "") != "table":
                continue
            ii = Recognizer.find_overlapped_with_threshold(b, rows, thr=0.3)
            if ii is not None:
                b["R"] = ii
                b["R_top"] = rows[ii]["top"]
                b["R_bott"] = rows[ii]["bottom"]

            ii = Recognizer.find_overlapped_with_threshold(b, headers, thr=0.3)
            if ii is not None:
                b["H_top"] = headers[ii]["top"]
                b["H_bott"] = headers[ii]["bottom"]
                b["H_left"] = headers[ii]["x0"]
                b["H_right"] = headers[ii]["x1"]
                b["H"] = ii

            ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
            if ii is not None:
                b["C"] = ii
                b["C_left"] = clmns[ii]["x0"]
                b["C_right"] = clmns[ii]["x1"]

            ii = Recognizer.find_overlapped_with_threshold(b, spans, thr=0.3)
            if ii is not None:
                b["H_top"] = spans[ii]["top"]
                b["H_bott"] = spans[ii]["bottom"]
                b["H_left"] = spans[ii]["x0"]
                b["H_right"] = spans[ii]["x1"]
                b["SP"] = ii

    def __ocr(self, pagenum, img, chars, ZM=3, device_id: int | None = None):
        start = timer()
        bxs = self.ocr.detect(np.array(img), device_id)
        logging.info(f"__ocr detecting boxes of a image cost ({timer() - start}s)")

        start = timer()
        if not bxs:
            self.boxes.append([])
            return
        bxs = [(line[0], line[1][0]) for line in bxs]
        bxs = Recognizer.sort_Y_firstly(
            [
                {"x0": b[0][0] / ZM, "x1": b[1][0] / ZM, "top": b[0][1] / ZM, "text": "", "txt": t, "bottom": b[-1][1] / ZM, "chars": [], "page_number": pagenum}
                for b, t in bxs
                if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]
            ],
            self.mean_height[pagenum - 1] / 3,
        )

        # merge chars in the same rect
        for c in chars:
            ii = Recognizer.find_overlapped(c, bxs)
            if ii is None:
                self.lefted_chars.append(c)
                continue
            ch = c["bottom"] - c["top"]
            bh = bxs[ii]["bottom"] - bxs[ii]["top"]
            if abs(ch - bh) / max(ch, bh) >= 0.7 and c["text"] != " ":
                self.lefted_chars.append(c)
                continue
            bxs[ii]["chars"].append(c)

        for b in bxs:
            if not b["chars"]:
                del b["chars"]
                continue
            m_ht = np.mean([c["height"] for c in b["chars"]])
            for c in Recognizer.sort_Y_firstly(b["chars"], m_ht):
                if c["text"] == " " and b["text"]:
                    if re.match(r"[0-9a-zA-Zа-яА-Я,.?;:!%%]", b["text"][-1]):
                        b["text"] += " "
                else:
                    b["text"] += c["text"]
            del b["chars"]

        logging.info(f"__ocr sorting {len(chars)} chars cost {timer() - start}s")
        start = timer()
        boxes_to_reg = []
        img_np = np.array(img)
        for b in bxs:
            if not b["text"]:
                left, right, top, bott = b["x0"] * ZM, b["x1"] * ZM, b["top"] * ZM, b["bottom"] * ZM
                b["box_image"] = self.ocr.get_rotate_crop_image(img_np, np.array([[left, top], [right, top], [right, bott], [left, bott]], dtype=np.float32))
                boxes_to_reg.append(b)
            del b["txt"]
        texts = self.ocr.recognize_batch([b["box_image"] for b in boxes_to_reg], device_id)
        for i in range(len(boxes_to_reg)):
            boxes_to_reg[i]["text"] = texts[i]
            del boxes_to_reg[i]["box_image"]
        logging.info(f"__ocr recognize {len(bxs)} boxes cost {timer() - start}s")
        bxs = [b for b in bxs if b["text"]]
        if self.mean_height[pagenum - 1] == 0:
            self.mean_height[pagenum - 1] = np.median([b["bottom"] - b["top"] for b in bxs])
        self.boxes.append(bxs)

    def _layouts_rec(self, ZM, drop=True):
        assert len(self.page_images) == len(self.boxes)
        self.boxes, self.page_layout = self.layouter(self.page_images, self.boxes, ZM, drop=drop)
        # cumlative Y
        for i in range(len(self.boxes)):
            self.boxes[i]["top"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["bottom"] += self.page_cum_height[self.boxes[i]["page_number"] - 1]

    def _text_merge(self):
        # merge adjusted boxes
        bxs = self.boxes

        def end_with(b, txt):
            txt = txt.strip()
            tt = b.get("text", "").strip()
            return tt and tt.find(txt) == len(tt) - len(txt)

        def start_with(b, txts):
            tt = b.get("text", "").strip()
            return tt and any([tt.find(t.strip()) == 0 for t in txts])

        # horizontally merge adjacent box with the same layout
        i = 0
        while i < len(bxs) - 1:
            b = bxs[i]
            b_ = bxs[i + 1]
            if b.get("layoutno", "0") != b_.get("layoutno", "1") or b.get("layout_type", "") in ["table", "figure", "equation"]:
                i += 1
                continue
            if abs(self._y_dis(b, b_)) < self.mean_height[bxs[i]["page_number"] - 1] / 3:
                # merge
                bxs[i]["x1"] = b_["x1"]
                bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                bxs[i]["text"] += b_["text"]
                bxs.pop(i + 1)
                continue
            i += 1
            continue

            dis_thr = 1
            dis = b["x1"] - b_["x0"]
            if b.get("layout_type", "") != "text" or b_.get("layout_type", "") != "text":
                if end_with(b, "，") or start_with(b_, "（，"):
                    dis_thr = -8
                else:
                    i += 1
                    continue

            if abs(self._y_dis(b, b_)) < self.mean_height[bxs[i]["page_number"] - 1] / 5 and dis >= dis_thr and b["x1"] < b_["x1"]:
                # merge
                bxs[i]["x1"] = b_["x1"]
                bxs[i]["top"] = (b["top"] + b_["top"]) / 2
                bxs[i]["bottom"] = (b["bottom"] + b_["bottom"]) / 2
                bxs[i]["text"] += b_["text"]
                bxs.pop(i + 1)
                continue
            i += 1
        self.boxes = bxs

    def _naive_vertical_merge(self):
        bxs = Recognizer.sort_Y_firstly(self.boxes, np.median(self.mean_height) / 3)
        i = 0
        while i + 1 < len(bxs):
            b = bxs[i]
            b_ = bxs[i + 1]
            if b["page_number"] < b_["page_number"] and re.match(r"[0-9  •一—-]+$", b["text"]):
                bxs.pop(i)
                continue
            if not b["text"].strip():
                bxs.pop(i)
                continue
            concatting_feats = [
                b["text"].strip()[-1] in ",;:'\"，、‘“；：-",
                len(b["text"].strip()) > 1 and b["text"].strip()[-2] in ",;:'\"，‘“、；：",
                b_["text"].strip() and b_["text"].strip()[0] in "。；？！?”）),，、：",
            ]
            # features for not concating
            feats = [
                b.get("layoutno", 0) != b_.get("layoutno", 0),
                b["text"].strip()[-1] in "。？！?",
                self.is_english and b["text"].strip()[-1] in ".!?",
                b["page_number"] == b_["page_number"] and b_["top"] - b["bottom"] > self.mean_height[b["page_number"] - 1] * 1.5,
                b["page_number"] < b_["page_number"] and abs(b["x0"] - b_["x0"]) > self.mean_width[b["page_number"] - 1] * 4,
            ]
            # split features
            detach_feats = [b["x1"] < b_["x0"], b["x0"] > b_["x1"]]
            if (any(feats) and not any(concatting_feats)) or any(detach_feats):
                logging.debug(
                    "{} {} {} {}".format(
                        b["text"],
                        b_["text"],
                        any(feats),
                        any(concatting_feats),
                    )
                )
                i += 1
                continue
            # merge up and down
            b["bottom"] = b_["bottom"]
            b["text"] += b_["text"]
            b["x0"] = min(b["x0"], b_["x0"])
            b["x1"] = max(b["x1"], b_["x1"])
            bxs.pop(i + 1)
        self.boxes = bxs

    def _concat_downward(self, concat_between_pages=True):
        self.boxes = Recognizer.sort_Y_firstly(self.boxes, 0)
        return

        # count boxes in the same row as a feature
        for i in range(len(self.boxes)):
            mh = self.mean_height[self.boxes[i]["page_number"] - 1]
            self.boxes[i]["in_row"] = 0
            j = max(0, i - 12)
            while j < min(i + 12, len(self.boxes)):
                if j == i:
                    j += 1
                    continue
                ydis = self._y_dis(self.boxes[i], self.boxes[j]) / mh
                if abs(ydis) < 1:
                    self.boxes[i]["in_row"] += 1
                elif ydis > 0:
                    break
                j += 1

        # concat between rows
        boxes = deepcopy(self.boxes)
        blocks = []
        while boxes:
            chunks = []

            def dfs(up, dp):
                chunks.append(up)
                i = dp
                while i < min(dp + 12, len(boxes)):
                    ydis = self._y_dis(up, boxes[i])
                    smpg = up["page_number"] == boxes[i]["page_number"]
                    mh = self.mean_height[up["page_number"] - 1]
                    mw = self.mean_width[up["page_number"] - 1]
                    if smpg and ydis > mh * 4:
                        break
                    if not smpg and ydis > mh * 16:
                        break
                    down = boxes[i]
                    if not concat_between_pages and down["page_number"] > up["page_number"]:
                        break

                    if up.get("R", "") != down.get("R", "") and up["text"][-1] != "，":
                        i += 1
                        continue

                    if re.match(r"[0-9]{2,3}/[0-9]{3}$", up["text"]) or re.match(r"[0-9]{2,3}/[0-9]{3}$", down["text"]) or not down["text"].strip():
                        i += 1
                        continue

                    if not down["text"].strip() or not up["text"].strip():
                        i += 1
                        continue

                    if up["x1"] < down["x0"] - 10 * mw or up["x0"] > down["x1"] + 10 * mw:
                        i += 1
                        continue

                    if i - dp < 5 and up.get("layout_type") == "text":
                        if up.get("layoutno", "1") == down.get("layoutno", "2"):
                            dfs(down, i + 1)
                            boxes.pop(i)
                            return
                        i += 1
                        continue

                    fea = self._updown_concat_features(up, down)
                    if self.updown_cnt_mdl.predict(xgb.DMatrix([fea]))[0] <= 0.5:
                        i += 1
                        continue
                    dfs(down, i + 1)
                    boxes.pop(i)
                    return

            dfs(boxes[0], 1)
            boxes.pop(0)
            if chunks:
                blocks.append(chunks)

        # concat within each block
        boxes = []
        for b in blocks:
            if len(b) == 1:
                boxes.append(b[0])
                continue
            t = b[0]
            for c in b[1:]:
                t["text"] = t["text"].strip()
                c["text"] = c["text"].strip()
                if not c["text"]:
                    continue
                if t["text"] and re.match(r"[0-9\.a-zA-Z]+$", t["text"][-1] + c["text"][-1]):
                    t["text"] += " "
                t["text"] += c["text"]
                t["x0"] = min(t["x0"], c["x0"])
                t["x1"] = max(t["x1"], c["x1"])
                t["page_number"] = min(t["page_number"], c["page_number"])
                t["bottom"] = c["bottom"]
                if not t["layout_type"] and c["layout_type"]:
                    t["layout_type"] = c["layout_type"]
            boxes.append(t)

        self.boxes = Recognizer.sort_Y_firstly(boxes, 0)

    def _filter_forpages(self):
        if not self.boxes:
            return
        findit = False
        i = 0
        while i < len(self.boxes):
            if not re.match(r"(contents|目录|目次|table of contents|致谢|acknowledge)$", re.sub(r"( | |\u3000)+", "", self.boxes[i]["text"].lower())):
                i += 1
                continue
            findit = True
            eng = re.match(r"[0-9a-zA-Z :'.-]{5,}", self.boxes[i]["text"].strip())
            self.boxes.pop(i)
            if i >= len(self.boxes):
                break
            prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            while not prefix:
                self.boxes.pop(i)
                if i >= len(self.boxes):
                    break
                prefix = self.boxes[i]["text"].strip()[:3] if not eng else " ".join(self.boxes[i]["text"].strip().split()[:2])
            self.boxes.pop(i)
            if i >= len(self.boxes) or not prefix:
                break
            for j in range(i, min(i + 128, len(self.boxes))):
                if not re.match(prefix, self.boxes[j]["text"]):
                    continue
                for k in range(i, j):
                    self.boxes.pop(i)
                break
        if findit:
            return

        page_dirty = [0] * len(self.page_images)
        for b in self.boxes:
            if re.search(r"(··|··|··)", b["text"]):
                page_dirty[b["page_number"] - 1] += 1
        page_dirty = set([i + 1 for i, t in enumerate(page_dirty) if t > 3])
        if not page_dirty:
            return
        i = 0
        while i < len(self.boxes):
            if self.boxes[i]["page_number"] in page_dirty:
                self.boxes.pop(i)
                continue
            i += 1

    def _merge_with_same_bullet(self):
        i = 0
        while i + 1 < len(self.boxes):
            b = self.boxes[i]
            b_ = self.boxes[i + 1]
            if not b["text"].strip():
                self.boxes.pop(i)
                continue
            if not b_["text"].strip():
                self.boxes.pop(i + 1)
                continue

            if (
                b["text"].strip()[0] != b_["text"].strip()[0]
                or b["text"].strip()[0].lower() in set("qwertyuopasdfghjklzxcvbnm")
                or rag_tokenizer.is_chinese(b["text"].strip()[0])
                or b["top"] > b_["bottom"]
            ):
                i += 1
                continue
            b_["text"] = b["text"] + "\n" + b_["text"]
            b_["x0"] = min(b["x0"], b_["x0"])
            b_["x1"] = max(b["x1"], b_["x1"])
            b_["top"] = b["top"]
            self.boxes.pop(i)

    def _extract_table_figure(self, need_image, ZM, return_html, need_position, separate_tables_figures=False):
        tables = {}
        figures = {}
        # extract figure and table boxes
        i = 0
        lst_lout_no = ""
        nomerge_lout_no = []
        while i < len(self.boxes):
            if "layoutno" not in self.boxes[i]:
                i += 1
                continue
            lout_no = str(self.boxes[i]["page_number"]) + "-" + str(self.boxes[i]["layoutno"])
            if TableStructureRecognizer.is_caption(self.boxes[i]) or self.boxes[i]["layout_type"] in ["table caption", "title", "figure caption", "reference"]:
                nomerge_lout_no.append(lst_lout_no)
            if self.boxes[i]["layout_type"] == "table":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in tables:
                    tables[lout_no] = []
                tables[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            if need_image and self.boxes[i]["layout_type"] == "figure":
                if re.match(r"(数据|资料|图表)*来源[:： ]", self.boxes[i]["text"]):
                    self.boxes.pop(i)
                    continue
                if lout_no not in figures:
                    figures[lout_no] = []
                figures[lout_no].append(self.boxes[i])
                self.boxes.pop(i)
                lst_lout_no = lout_no
                continue
            i += 1

        # merge table on different pages
        nomerge_lout_no = set(nomerge_lout_no)
        tbls = sorted([(k, bxs) for k, bxs in tables.items()], key=lambda x: (x[1][0]["top"], x[1][0]["x0"]))

        i = len(tbls) - 1
        while i - 1 >= 0:
            k0, bxs0 = tbls[i - 1]
            k, bxs = tbls[i]
            i -= 1
            if k0 in nomerge_lout_no:
                continue
            if bxs[0]["page_number"] == bxs0[0]["page_number"]:
                continue
            if bxs[0]["page_number"] - bxs0[0]["page_number"] > 1:
                continue
            mh = self.mean_height[bxs[0]["page_number"] - 1]
            if self._y_dis(bxs0[-1], bxs[0]) > mh * 23:
                continue
            tables[k0].extend(tables[k])
            del tables[k]

        def x_overlapped(a, b):
            return not any([a["x1"] < b["x0"], a["x0"] > b["x1"]])

        # find captions and pop out
        i = 0
        while i < len(self.boxes):
            c = self.boxes[i]
            # mh = self.mean_height[c["page_number"]-1]
            if not TableStructureRecognizer.is_caption(c):
                i += 1
                continue

            # find the nearest layouts
            def nearest(tbls):
                nonlocal c
                mink = ""
                minv = 1000000000
                for k, bxs in tbls.items():
                    for b in bxs:
                        if b.get("layout_type", "").find("caption") >= 0:
                            continue
                        y_dis = self._y_dis(c, b)
                        x_dis = self._x_dis(c, b) if not x_overlapped(c, b) else 0
                        dis = y_dis * y_dis + x_dis * x_dis
                        if dis < minv:
                            mink = k
                            minv = dis
                return mink, minv

            tk, tv = nearest(tables)
            fk, fv = nearest(figures)
            # if min(tv, fv) > 2000:
            #    i += 1
            #    continue
            if tv < fv and tk:
                tables[tk].insert(0, c)
                logging.debug("TABLE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            elif fk:
                figures[fk].insert(0, c)
                logging.debug("FIGURE:" + self.boxes[i]["text"] + "; Cap: " + tk)
            self.boxes.pop(i)

        def cropout(bxs, ltype, poss):
            nonlocal ZM
            pn = set([b["page_number"] - 1 for b in bxs])
            if len(pn) < 2:
                pn = list(pn)[0]
                ht = self.page_cum_height[pn]
                b = {"x0": np.min([b["x0"] for b in bxs]), "top": np.min([b["top"] for b in bxs]) - ht, "x1": np.max([b["x1"] for b in bxs]), "bottom": np.max([b["bottom"] for b in bxs]) - ht}
                louts = [layout for layout in self.page_layout[pn] if layout["type"] == ltype]
                ii = Recognizer.find_overlapped(b, louts, naive=True)
                if ii is not None:
                    b = louts[ii]
                else:
                    logging.warning(f"Missing layout match: {pn + 1},%s" % (bxs[0].get("layoutno", "")))

                left, top, right, bott = b["x0"], b["top"], b["x1"], b["bottom"]
                if right < left:
                    right = left + 1
                poss.append((pn + self.page_from, left, right, top, bott))
                return self.page_images[pn].crop((left * ZM, top * ZM, right * ZM, bott * ZM))
            pn = {}
            for b in bxs:
                p = b["page_number"] - 1
                if p not in pn:
                    pn[p] = []
                pn[p].append(b)
            pn = sorted(pn.items(), key=lambda x: x[0])
            imgs = [cropout(arr, ltype, poss) for p, arr in pn]
            pic = Image.new("RGB", (int(np.max([i.size[0] for i in imgs])), int(np.sum([m.size[1] for m in imgs]))), (245, 245, 245))
            height = 0
            for img in imgs:
                pic.paste(img, (0, int(height)))
                height += img.size[1]
            return pic

        res = []
        positions = []
        figure_results = []
        figure_positions = []
        # crop figure out and add caption
        for k, bxs in figures.items():
            txt = "\n".join([b["text"] for b in bxs])
            if not txt:
                continue

            poss = []

            if separate_tables_figures:
                figure_results.append((cropout(bxs, "figure", poss), [txt]))
                figure_positions.append(poss)
            else:
                res.append((cropout(bxs, "figure", poss), [txt]))
                positions.append(poss)

        for k, bxs in tables.items():
            if not bxs:
                continue
            bxs = Recognizer.sort_Y_firstly(bxs, np.mean([(b["bottom"] - b["top"]) / 2 for b in bxs]))

            poss = []

            res.append((cropout(bxs, "table", poss), self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
            positions.append(poss)

        if separate_tables_figures:
            assert len(positions) + len(figure_positions) == len(res) + len(figure_results)
            if need_position:
                return list(zip(res, positions)), list(zip(figure_results, figure_positions))
            else:
                return res, figure_results
        else:
            assert len(positions) == len(res)
            if need_position:
                return list(zip(res, positions))
            else:
                return res

    def proj_match(self, line):
        if len(line) <= 2:
            return
        if re.match(r"[0-9 ().,%%+/-]+$", line):
            return False
        for p, j in [
            (r"第[零一二三四五六七八九十百]+章", 1),
            (r"第[零一二三四五六七八九十百]+[条节]", 2),
            (r"[零一二三四五六七八九十百]+[、 　]", 3),
            (r"[\(（][零一二三四五六七八九十百]+[）\)]", 4),
            (r"[0-9]+(、|\.[　 ]|\.[^0-9])", 5),
            (r"[0-9]+\.[0-9]+(、|[. 　]|[^0-9])", 6),
            (r"[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 7),
            (r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(、|[ 　]|[^0-9])", 8),
            (r".{,48}[：:?？]$", 9),
            (r"[0-9]+）", 10),
            (r"[\(（][0-9]+[）\)]", 11),
            (r"[零一二三四五六七八九十百]+是", 12),
            (r"[⚫•➢✓]", 12),
        ]:
            if re.match(p, line):
                return j
        return

    def _line_tag(self, bx, ZM):
        pn = [bx["page_number"]]
        top = bx["top"] - self.page_cum_height[pn[0] - 1]
        bott = bx["bottom"] - self.page_cum_height[pn[0] - 1]
        page_images_cnt = len(self.page_images)
        if pn[-1] - 1 >= page_images_cnt:
            return ""
        while bott * ZM > self.page_images[pn[-1] - 1].size[1]:
            bott -= self.page_images[pn[-1] - 1].size[1] / ZM
            pn.append(pn[-1] + 1)
            if pn[-1] - 1 >= page_images_cnt:
                return ""

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), bx["x0"], bx["x1"], top, bott)

    def __filterout_scraps(self, boxes, ZM):
        def width(b):
            return b["x1"] - b["x0"]

        def height(b):
            return b["bottom"] - b["top"]

        def usefull(b):
            if b.get("layout_type"):
                return True
            if width(b) > self.page_images[b["page_number"] - 1].size[0] / ZM / 3:
                return True
            if b["bottom"] - b["top"] > self.mean_height[b["page_number"] - 1]:
                return True
            return False

        res = []
        while boxes:
            lines = []
            widths = []
            pw = self.page_images[boxes[0]["page_number"] - 1].size[0] / ZM
            mh = self.mean_height[boxes[0]["page_number"] - 1]
            mj = self.proj_match(boxes[0]["text"]) or boxes[0].get("layout_type", "") == "title"

            def dfs(line, st):
                nonlocal mh, pw, lines, widths
                lines.append(line)
                widths.append(width(line))
                mmj = self.proj_match(line["text"]) or line.get("layout_type", "") == "title"
                for i in range(st + 1, min(st + 20, len(boxes))):
                    if (boxes[i]["page_number"] - line["page_number"]) > 0:
                        break
                    if not mmj and self._y_dis(line, boxes[i]) >= 3 * mh and height(line) < 1.5 * mh:
                        break

                    if not usefull(boxes[i]):
                        continue
                    if mmj or (self._x_dis(boxes[i], line) < pw / 10):
                        # and abs(width(boxes[i])-width_mean)/max(width(boxes[i]),width_mean)<0.5):
                        # concat following
                        dfs(boxes[i], i)
                        boxes.pop(i)
                        break

            try:
                if usefull(boxes[0]):
                    dfs(boxes[0], 0)
                else:
                    logging.debug("WASTE: " + boxes[0]["text"])
            except Exception:
                pass
            boxes.pop(0)
            mw = np.mean(widths)
            if mj or mw / pw >= 0.35 or mw > 200:
                res.append("\n".join([c["text"] + self._line_tag(c, ZM) for c in lines]))
            else:
                logging.debug("REMOVED: " + "<<".join([c["text"] for c in lines]))

        return "\n\n".join(res)

    @staticmethod
    def total_page_number(fnm, binary=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                pdf = pdfplumber.open(fnm) if not binary else pdfplumber.open(BytesIO(binary))
            total_page = len(pdf.pages)
            pdf.close()
            return total_page
        except Exception:
            logging.exception("total_page_number")

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        self.lefted_chars = []
        self.mean_height = []
        self.mean_width = []
        self.boxes = []
        self.garbages = {}
        self.page_cum_height = [0]
        self.page_layout = []
        self.page_from = page_from
        start = timer()
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                with pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm)) as pdf:
                    self.pdf = pdf
                    self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]

                    try:
                        self.page_chars = [[c for c in page.dedupe_chars().chars if self._has_color(c)] for page in self.pdf.pages[page_from:page_to]]
                    except Exception as e:
                        logging.warning(f"Failed to extract characters for pages {page_from}-{page_to}: {str(e)}")
                        self.page_chars = [[] for _ in range(page_to - page_from)]  # If failed to extract, using empty list instead.

                    self.total_page = len(self.pdf.pages)

        except Exception:
            logging.exception("RAGFlowPdfParser __images__")
        logging.info(f"__images__ dedupe_chars cost {timer() - start}s")

        self.outlines = []
        try:
            with pdf2_read(fnm if isinstance(fnm, str) else BytesIO(fnm)) as pdf:
                self.pdf = pdf

                outlines = self.pdf.outline

                def dfs(arr, depth):
                    for a in arr:
                        if isinstance(a, dict):
                            self.outlines.append((a["/Title"], depth))
                            continue
                        dfs(a, depth + 1)

                dfs(outlines, 0)

        except Exception as e:
            logging.warning(f"Outlines exception: {e}")

        if not self.outlines:
            logging.warning("Miss outlines")

        logging.debug("Images converted.")
        self.is_english = [
            re.search(r"[a-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join(random.choices([c["text"] for c in self.page_chars[i]], k=min(100, len(self.page_chars[i])))))
            for i in range(len(self.page_chars))
        ]
        if sum([1 if e else 0 for e in self.is_english]) > len(self.page_images) / 2:
            self.is_english = True
        else:
            self.is_english = False

        async def __img_ocr(i, id, img, chars, limiter):
            j = 0
            while j + 1 < len(chars):
                if (
                    chars[j]["text"]
                    and chars[j + 1]["text"]
                    and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
                    and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                ):
                    chars[j]["text"] += " "
                j += 1

            if limiter:
                async with limiter:
                    await trio.to_thread.run_sync(lambda: self.__ocr(i + 1, img, chars, zoomin, id))
            else:
                self.__ocr(i + 1, img, chars, zoomin, id)

            if callback and i % 6 == 5:
                callback(prog=(i + 1) * 0.6 / len(self.page_images), msg="")

        async def __img_ocr_launcher():
            def __ocr_preprocess(page_idx, img):
                chars = self.page_chars[page_idx] if not self.is_english else []
                self.mean_height.append(np.median(sorted([c["height"] for c in chars])) if chars else 0)
                self.mean_width.append(np.median(sorted([c["width"] for c in chars])) if chars else 8)
                self.page_cum_height.append(img.size[1] / zoomin)
                return chars
            
            # Pipeline 数据结构
            from dataclasses import dataclass
            from typing import Optional
            
            @dataclass
            class PreparedPage:
                """S1 预处理后的页面数据"""
                page_idx: int
                img: Image.Image
                chars: list
                zoomin: int
            
            @dataclass
            class OCRResult:
                """S2 GPU 推理后的结果"""
                page_idx: int
                pagenum: int  # 1-based page number
                boxes: list
                lefted_chars: list
                mean_height_val: float

            total_pages = len(self.page_images)
            print(f"[CONCURRENCY DEBUG] Starting OCR launcher for {total_pages} pages")
            print(f"[CONCURRENCY DEBUG] parallel_limiter = {self.parallel_limiter}")
            print(f"[CONCURRENCY DEBUG] PARALLEL_DEVICES = {PARALLEL_DEVICES}")
            
            # Check if multi-session mode is enabled (for single GPU)
            gpu_sessions = int(os.environ.get("DEEPDOC_GPU_SESSIONS", "0"))
            use_multi_session = gpu_sessions > 0 and PARALLEL_DEVICES > 0
            if use_multi_session:
                print(f"[CONCURRENCY DEBUG] Multi-session mode enabled: {gpu_sessions} sessions on GPU 0")
                # In multi-session mode, use session_id as index (0 to gpu_sessions-1)
                # All sessions use device_id=0, but have different session_id
                num_parallel = gpu_sessions
            else:
                # In multi-GPU mode, use device_id as index (0 to PARALLEL_DEVICES-1)
                num_parallel = PARALLEL_DEVICES

            if self.parallel_limiter:
                processing_mode = os.environ.get("DEEPDOC_PROCESSING_MODE", "default")
                print(f"[CONCURRENCY DEBUG] Processing mode: {processing_mode}")

                if processing_mode == "multiprocess":
                    # 方案E: 多进程处理
                    print(f"[CONCURRENCY DEBUG] Using MULTIPROCESS mode")
                    import multiprocessing
                    
                    # 准备任务列表：先预处理所有页面
                    tasks = []
                    for i, img in enumerate(self.page_images):
                        chars = __ocr_preprocess(i, img)
                        
                        # 处理chars中的空格合并逻辑（与__img_ocr中的逻辑一致）
                        j = 0
                        while j + 1 < len(chars):
                            if (
                                chars[j]["text"]
                                and chars[j + 1]["text"]
                                and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
                                and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                            ):
                                chars[j]["text"] += " "
                            j += 1
                        
                        # 获取当前mean_height（预处理后已填充）
                        mean_height_val = self.mean_height[i] if i < len(self.mean_height) else 0
                        
                        # 将图像转换为可序列化的格式（PNG字节）
                        img_bytes = BytesIO()
                        img.save(img_bytes, format='PNG')
                        img_bytes_data = img_bytes.getvalue()
                        
                        # 计算parallel_id (device_id in multi-GPU mode, session_id in multi-session mode)
                        parallel_id_val = i % num_parallel
                        
                        tasks.append((i, img_bytes_data, chars, zoomin, parallel_id_val, mean_height_val))
                    
                    # 使用进程池处理
                    num_workers = min(PARALLEL_DEVICES, total_pages)
                    print(f"[CONCURRENCY DEBUG] Starting multiprocess pool with {num_workers} workers")
                    with multiprocessing.Pool(processes=num_workers) as pool:
                        # 使用模块级别的函数，可以被pickle序列化
                        results = pool.map(_multiprocess_page_worker_standalone, tasks)
                    
                    # 按page_idx排序结果（确保顺序正确）
                    results.sort(key=lambda x: x[0])
                    
                    # 整合结果到主进程的数据结构
                    for page_idx, boxes, mean_height_val, lefted_chars in results:
                        # 更新mean_height（如果之前为0）
                        if page_idx < len(self.mean_height) and self.mean_height[page_idx] == 0:
                            self.mean_height[page_idx] = mean_height_val
                        # 添加boxes
                        self.boxes.append(boxes)
                        # 合并lefted_chars到主进程
                        self.lefted_chars.extend(lefted_chars)
                    
                    print(f"[CONCURRENCY DEBUG] Multiprocess completed {len(results)} pages")
                    
                elif processing_mode == "fixed_batch":
                    # 方案C: 固定批处理
                    batch_size = int(os.environ.get("DEEPDOC_BATCH_SIZE", str(PARALLEL_DEVICES * 2)))
                    print(f"[CONCURRENCY DEBUG] Using FIXED_BATCH mode, batch_size={batch_size}")
                    
                    async with trio.open_nursery() as nursery:
                        for batch_start in range(0, total_pages, batch_size):
                            batch_end = min(batch_start + batch_size, total_pages)
                            batch_indices = list(range(batch_start, batch_end))
                            
                            print(f"[CONCURRENCY DEBUG] Processing batch {batch_start//batch_size + 1}: pages {batch_start+1}-{batch_end}")
                            
                            async def process_batch(batch_indices):
                                async with trio.open_nursery() as batch_nursery:
                                    for i in batch_indices:
                                        img = self.page_images[i]
                                        # 使用 trio.to_thread.run_sync 包装同步预处理操作，避免阻塞事件循环
                                        chars = await trio.to_thread.run_sync(__ocr_preprocess, i, img)
                                        parallel_id = i % num_parallel
                                        batch_nursery.start_soon(__img_ocr, i, parallel_id, img, chars, self.parallel_limiter[parallel_id])
                            
                            await process_batch(batch_indices)
                    
                elif processing_mode == "sliding_batch":
                    # 方案D: 滑动窗口批处理
                    batch_size = int(os.environ.get("DEEPDOC_BATCH_SIZE", str(PARALLEL_DEVICES * 2)))
                    max_concurrent_batches = int(os.environ.get("DEEPDOC_MAX_CONCURRENT_BATCHES", "2"))
                    print(f"[CONCURRENCY DEBUG] Using SLIDING_BATCH mode, batch_size={batch_size}, max_concurrent_batches={max_concurrent_batches}")
                    
                    # 创建页面锁机制，确保每个页面只处理一次（包括预处理和OCR）
                    # 为每个页面创建一个锁，避免并发时重复处理
                    page_locks = {i: trio.Lock() for i in range(total_pages)}
                    processed_pages = set()  # 跟踪已完全处理的页面索引（包括预处理和OCR）
                    
                    # 创建批处理限制器
                    batch_limiter = trio.CapacityLimiter(max_concurrent_batches)
                    
                    async def process_page_safely(page_idx):
                        """安全地处理单个页面：确保每个页面只处理一次（预处理+OCR）"""
                        img = self.page_images[page_idx]
                        
                        # 使用页面锁确保每个页面只被处理一次
                        async with page_locks[page_idx]:
                            if page_idx not in processed_pages:
                                # 页面尚未处理，执行完整的处理流程
                                # 1. 预处理（使用线程池避免阻塞事件循环）
                                chars = await trio.to_thread.run_sync(__ocr_preprocess, page_idx, img)
                                
                                # 2. OCR处理
                                parallel_id = page_idx % num_parallel
                                await __img_ocr(page_idx, parallel_id, img, chars, self.parallel_limiter[parallel_id])
                                
                                # 3. 标记为已处理
                                processed_pages.add(page_idx)
                            # 如果页面已处理，直接跳过（不做任何操作）
                    
                    async def process_sliding_batch(batch_start):
                        async with batch_limiter:
                            batch_end = min(batch_start + batch_size, total_pages)
                            batch_indices = list(range(batch_start, batch_end))
                            
                            print(f"[CONCURRENCY DEBUG] Processing sliding batch: pages {batch_start+1}-{batch_end}")
                            
                            # 并发处理批次中的所有页面
                            async with trio.open_nursery() as batch_nursery:
                                for i in batch_indices:
                                    batch_nursery.start_soon(process_page_safely, i)
                    
                    async with trio.open_nursery() as nursery:
                        for batch_start in range(0, total_pages, batch_size // 2):  # 滑动窗口，步长为 batch_size/2
                            nursery.start_soon(process_sliding_batch, batch_start)
                    
                elif processing_mode == "optimized_batch_serial":
                    # 方案H: 优化批处理 - 批次内串行
                    BATCH_SIZE = int(os.environ.get("DEEPDOC_BATCH_SIZE", "16"))
                    print(f"[CONCURRENCY DEBUG] Using OPTIMIZED_BATCH_SERIAL mode, batch_size={BATCH_SIZE}")
                    
                    # 先预处理所有页面（串行执行，确保顺序正确）
                    print(f"[CONCURRENCY DEBUG] Preprocessing all {total_pages} pages...")
                    preprocessed_chars = {}
                    for i, img in enumerate(self.page_images):
                        chars = __ocr_preprocess(i, img)
                        preprocessed_chars[i] = chars
                    print(f"[CONCURRENCY DEBUG] Preprocessing completed for all pages")
                    
                    def process_batch_sync(batch_indices):
                        """在线程池中一次性处理一批：OCR（批次内串行）"""
                        for i in batch_indices:
                            img = self.page_images[i]
                            chars = preprocessed_chars[i]  # 使用预处理好的chars
                            
                            # 字符合并逻辑（与 __img_ocr 中的逻辑一致）
                            j = 0
                            while j + 1 < len(chars):
                                if (
                                    chars[j]["text"]
                                    and chars[j + 1]["text"]
                                    and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
                                    and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                                ):
                                    chars[j]["text"] += " "
                                j += 1
                            
                            # 执行 OCR（核心计算）
                            # In multi-session mode, use session_id as index; otherwise use device_id
                            parallel_id = i % num_parallel
                            self.__ocr(i + 1, img, chars, zoomin, parallel_id)
                    
                    async def run_batch_task(batch_indices, limiter):
                        """异步包装器：将批次任务提交到线程池"""
                        async with limiter:
                            await trio.to_thread.run_sync(process_batch_sync, batch_indices)
                    
                    # 主调度循环（极速分发模式）
                    print(f"[CONCURRENCY DEBUG] Starting optimized batch processing. Total: {total_pages}, Batch: {BATCH_SIZE}")
                    concurrent_start = timer()
                    
                    async with trio.open_nursery() as nursery:
                        # 按批次切分任务，快速分发所有批次
                        for start_idx in range(0, total_pages, BATCH_SIZE):
                            end_idx = min(start_idx + BATCH_SIZE, total_pages)
                            batch_indices = list(range(start_idx, end_idx))
                            
                            # 负载均衡：根据批次号分配 Limiter
                            limiter = self.parallel_limiter[start_idx % num_parallel]
                            
                            # 非阻塞分发：瞬间完成所有批次的任务投递
                            nursery.start_soon(run_batch_task, batch_indices, limiter)
                    
                    concurrent_elapsed = timer() - concurrent_start
                    print(f"[CONCURRENCY DEBUG] All optimized batch tasks completed in {concurrent_elapsed:.3f}s")
                    
                elif processing_mode == "pipeline":
                    # 方案J: 三阶段流水线模式（CPU-GPU 协同）
                    # 使用外层定义的 gpu_sessions 和 use_multi_session
                    pipeline_gpu_sessions = int(os.environ.get("DEEPDOC_GPU_SESSIONS", "1"))
                    if pipeline_gpu_sessions <= 0:
                        pipeline_gpu_sessions = 1
                    
                    # 确保 pipeline 模式下的 session 数与 OCR 初始化时一致
                    if use_multi_session:
                        # 多 session 模式：使用 session_id 作为 parallel_id
                        pipeline_num_parallel = gpu_sessions
                    else:
                        # 多 GPU 模式：使用 device_id 作为 parallel_id
                        pipeline_num_parallel = num_parallel
                    
                    # 配置参数
                    S1_WORKERS = int(os.environ.get("DEEPDOC_PIPELINE_S1_WORKERS", "8"))
                    S3_WORKERS = int(os.environ.get("DEEPDOC_PIPELINE_S3_WORKERS", "2"))
                    QUEUE_CAPACITY = int(os.environ.get("DEEPDOC_PIPELINE_QUEUE_CAPACITY", "4"))
                    
                    print(f"[PIPELINE] Starting pipeline mode: S1_workers={S1_WORKERS}, S2_sessions={pipeline_gpu_sessions}, S3_workers={S3_WORKERS}, queue_capacity={QUEUE_CAPACITY}")
                    print(f"[PIPELINE] use_multi_session={use_multi_session}, pipeline_num_parallel={pipeline_num_parallel}")
                    
                    # 提前分配 mean_height, mean_width, page_cum_height 列表（确保线程安全）
                    self.mean_height = [0.0] * total_pages
                    self.mean_width = [8.0] * total_pages
                    self.page_cum_height = [0.0] * (total_pages + 1)
                    # 注意：不预分配 self.boxes，因为 __ocr 使用 append
                    # 我们会在 pipeline 结束时对 self.boxes 进行排序
                    self.boxes = []
                    
                    # 创建队列
                    queue_prepared_send, queue_prepared_recv = trio.open_memory_channel(QUEUE_CAPACITY)
                    queue_inferred_send, queue_inferred_recv = trio.open_memory_channel(QUEUE_CAPACITY)
                    
                    # 页面锁：确保每个页面只被 S2 处理一次
                    page_locks = [trio.Lock() for _ in range(total_pages)]
                    
                    # 用于跟踪 S1 完成状态（所有页面预处理完成）
                    s1_completed = trio.Event()
                    s1_pages_processed = 0
                    s1_lock = trio.Lock()
                    
                    async def s1_preprocess_worker(worker_id: int):
                        """S1 预处理 Worker：CPU 密集型"""
                        nonlocal s1_pages_processed
                        processed_count = 0
                        
                        for page_idx in range(total_pages):
                            # 负载均衡：每个 worker 处理一部分页面
                            if page_idx % S1_WORKERS != worker_id:
                                continue
                            
                            img = self.page_images[page_idx]
                            
                            # 预处理（在线程池中执行，避免阻塞事件循环）
                            def do_preprocess():
                                chars = __ocr_preprocess(page_idx, img)
                                
                                # 字符合并逻辑（与 __img_ocr 中的逻辑一致）
                                j = 0
                                while j + 1 < len(chars):
                                    if (
                                        chars[j]["text"]
                                        and chars[j + 1]["text"]
                                        and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
                                        and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                                    ):
                                        chars[j]["text"] += " "
                                    j += 1
                                
                                return PreparedPage(
                                    page_idx=page_idx,
                                    img=img,
                                    chars=chars,
                                    zoomin=zoomin
                                )
                            
                            prepared_page = await trio.to_thread.run_sync(do_preprocess)
                            
                            # 发送到 S2 队列
                            await queue_prepared_send.send(prepared_page)
                            processed_count += 1
                            
                            # 更新完成计数
                            async with s1_lock:
                                s1_pages_processed += 1
                                if s1_pages_processed >= total_pages:
                                    s1_completed.set()
                        
                        print(f"[PIPELINE] S1 worker {worker_id} completed {processed_count} pages")
                    
                    async def s2_inference_worker(session_id: int):
                        """S2 GPU 推理 Worker：每个 session 一个 worker"""
                        processed_count = 0
                        # 计算 parallel_id：在多 session 模式下使用 session_id，在多 GPU 模式下使用 device_id
                        if use_multi_session:
                            parallel_id = session_id  # 多 session 模式：session_id 直接作为 parallel_id
                            # 验证 parallel_id 在有效范围内
                            if parallel_id >= len(self.ocr.text_detector):
                                raise ValueError(f"parallel_id {parallel_id} out of range (max: {len(self.ocr.text_detector) - 1})")
                        else:
                            parallel_id = session_id % pipeline_num_parallel  # 多 GPU 模式：使用 device_id
                            # 验证 parallel_id 在有效范围内
                            if parallel_id >= len(self.ocr.text_detector):
                                raise ValueError(f"parallel_id {parallel_id} out of range (max: {len(self.ocr.text_detector) - 1})")
                        
                        print(f"[PIPELINE] S2 worker {session_id} started with parallel_id={parallel_id} (detector_count={len(self.ocr.text_detector)}, recognizer_count={len(self.ocr.text_recognizer)})")
                        
                        try:
                            async for prepared_page in queue_prepared_recv:
                                page_idx = prepared_page.page_idx
                                pagenum = page_idx + 1
                                
                                # 使用页面锁确保每个页面只被处理一次
                                async with page_locks[page_idx]:
                                    # GPU 推理（在线程池中执行，但主要耗时在 GPU）
                                    def do_inference():
                                        # 执行 OCR（__ocr 会直接修改 self.boxes 和 self.lefted_chars）
                                        # 注意：__ocr 使用 append，所以新 boxes 会添加到 self.boxes 末尾
                                        self.__ocr(pagenum, prepared_page.img, prepared_page.chars, prepared_page.zoomin, parallel_id)
                                        
                                        mean_height_val = self.mean_height[page_idx] if page_idx < len(self.mean_height) else 0.0
                                        
                                        return OCRResult(
                                            page_idx=page_idx,
                                            pagenum=pagenum,
                                            boxes=[],  # boxes 已在 __ocr 中添加到 self.boxes
                                            lefted_chars=[],
                                            mean_height_val=mean_height_val
                                        )
                                    
                                    ocr_result = await trio.to_thread.run_sync(do_inference)
                                    
                                    # 发送到 S3 队列
                                    await queue_inferred_send.send(ocr_result)
                                    processed_count += 1
                        except trio.EndOfChannel:
                            pass  # 正常结束
                        
                        print(f"[PIPELINE] S2 worker (session {session_id}) completed {processed_count} pages")
                    
                    async def s3_postprocess_worker(worker_id: int):
                        """S3 后处理 Worker：CPU 密集型（验证和清理）"""
                        processed_count = 0
                        
                        try:
                            async for ocr_result in queue_inferred_recv:
                                # 后处理：验证数据完整性
                                # 由于 __ocr 已经直接修改了 self.boxes 和 self.lefted_chars，
                                # 这里主要是验证和记录完成状态
                                
                                # 验证 boxes 已正确添加（__ocr 使用 append，所以顺序可能不是按 page_idx）
                                # 我们只需要确保所有页面都被处理即可
                                processed_count += 1
                        except trio.EndOfChannel:
                            pass  # 正常结束
                        
                        print(f"[PIPELINE] S3 worker {worker_id} completed {processed_count} pages")
                    
                    # 启动所有 worker
                    async with trio.open_nursery() as nursery:
                        # 启动 S1 workers
                        for worker_id in range(S1_WORKERS):
                            nursery.start_soon(s1_preprocess_worker, worker_id)
                        
                        # 启动 S2 workers（每个 session 一个）
                        for session_id in range(pipeline_gpu_sessions):
                            nursery.start_soon(s2_inference_worker, session_id)
                        
                        # 启动 S3 workers
                        for worker_id in range(S3_WORKERS):
                            nursery.start_soon(s3_postprocess_worker, worker_id)
                    
                    # 等待所有 S1 worker 完成
                    await s1_completed.wait()
                    
                    # 关闭发送端，通知接收端结束
                    await queue_prepared_send.aclose()
                    await queue_inferred_send.aclose()
                    
                    # 等待所有 worker 完成（nursery 会自动等待）
                    # 注意：这里 nursery 已经关闭，所有 worker 应该已经完成
                    
                    # 对 self.boxes 进行排序，确保顺序与 page_images 一致
                    # __ocr 使用 append，所以 self.boxes 的顺序可能不是按 page_idx
                    # 我们需要按照 page_number 对 self.boxes 进行分组和排序
                    if len(self.boxes) > 0:
                        # 按 page_number 分组 boxes
                        boxes_by_page = {}
                        for box_list in self.boxes:
                            if box_list and len(box_list) > 0:
                                # 获取第一个 box 的 page_number（同一页面的 boxes 应该有相同的 page_number）
                                page_num = box_list[0].get("page_number", 0)
                                if page_num > 0:
                                    page_idx = page_num - 1
                                    boxes_by_page[page_idx] = box_list
                        
                        # 按 page_idx 顺序重建 self.boxes
                        sorted_boxes = []
                        for page_idx in range(total_pages):
                            sorted_boxes.append(boxes_by_page.get(page_idx, []))
                        
                        self.boxes = sorted_boxes
                    
                    print(f"[PIPELINE] All pipeline workers completed, boxes sorted")
                    
                elif processing_mode == "optimized_batch_parallel":
                    # 方案I: 优化批处理 - 批次内并行
                    BATCH_SIZE = int(os.environ.get("DEEPDOC_BATCH_SIZE", "16"))
                    print(f"[CONCURRENCY DEBUG] Using OPTIMIZED_BATCH_PARALLEL mode, batch_size={BATCH_SIZE}")
                    
                    # 先预处理所有页面（串行执行，确保顺序正确）
                    print(f"[CONCURRENCY DEBUG] Preprocessing all {total_pages} pages...")
                    preprocessed_chars = {}
                    for i, img in enumerate(self.page_images):
                        chars = __ocr_preprocess(i, img)
                        preprocessed_chars[i] = chars
                    print(f"[CONCURRENCY DEBUG] Preprocessing completed for all pages")
                    
                    async def process_batch_parallel(batch_indices, limiter):
                        """批次内并行处理：OCR"""
                        async with trio.open_nursery() as batch_nursery:
                            for i in batch_indices:
                                img = self.page_images[i]
                                chars = preprocessed_chars[i]  # 使用预处理好的chars
                                
                                # 字符合并逻辑
                                j = 0
                                while j + 1 < len(chars):
                                    if (
                                        chars[j]["text"]
                                        and chars[j + 1]["text"]
                                        and re.match(r"[0-9a-zA-Z,.:;!%]+", chars[j]["text"] + chars[j + 1]["text"])
                                        and chars[j + 1]["x0"] - chars[j]["x1"] >= min(chars[j + 1]["width"], chars[j]["width"]) / 2
                                    ):
                                        chars[j]["text"] += " "
                                    j += 1
                                
                                # OCR 处理（在线程池中，受 limiter 控制）
                                parallel_id = i % num_parallel
                                async with limiter:
                                    await trio.to_thread.run_sync(lambda: self.__ocr(i + 1, img, chars, zoomin, parallel_id))
                    
                    async def run_batch_task_parallel(batch_indices, batch_limiter):
                        """批次任务包装器"""
                        async with batch_limiter:
                            await process_batch_parallel(batch_indices, self.parallel_limiter[0])  # 使用第一个 limiter
                    
                    # 主调度循环（极速分发模式）
                    print(f"[CONCURRENCY DEBUG] Starting optimized batch parallel processing. Total: {total_pages}, Batch: {BATCH_SIZE}")
                    concurrent_start = timer()
                    
                    # 批次级别的并发控制
                    max_concurrent_batches = PARALLEL_DEVICES
                    batch_limiter = trio.CapacityLimiter(max_concurrent_batches)
                    
                    async with trio.open_nursery() as nursery:
                        # 按批次切分任务，快速分发所有批次
                        for start_idx in range(0, total_pages, BATCH_SIZE):
                            end_idx = min(start_idx + BATCH_SIZE, total_pages)
                            batch_indices = list(range(start_idx, end_idx))
                            
                            # 非阻塞分发：瞬间完成所有批次的任务投递
                            nursery.start_soon(run_batch_task_parallel, batch_indices, batch_limiter)
                    
                    concurrent_elapsed = timer() - concurrent_start
                    print(f"[CONCURRENCY DEBUG] All optimized batch parallel tasks completed in {concurrent_elapsed:.3f}s")
                    
                else:
                    # 方案A/B: 默认并行模式（原始 trio 并发）
                    print(f"[CONCURRENCY DEBUG] Using PARALLEL mode with {len(self.parallel_limiter)} limiters")
                    concurrent_start = timer()
                    print(f"[CONCURRENCY DEBUG] Concurrent start time: {concurrent_start:.3f}")
                    
                    # 对于大文档（>100页），移除调度延迟以避免累积阻塞
                    # 小文档保留小延迟以避免内存峰值
                    use_delay = len(self.page_images) < 100
                    
                    async with trio.open_nursery() as nursery:
                        for i, img in enumerate(self.page_images):
                            # 使用 trio.to_thread.run_sync 包装同步预处理操作，避免阻塞事件循环
                            chars = await trio.to_thread.run_sync(__ocr_preprocess, i, img)
                            parallel_id = i % num_parallel
                            print(f"[CONCURRENCY DEBUG] Scheduling page {i+1}/{total_pages} to parallel_id={parallel_id} at {timer():.3f}")
                            nursery.start_soon(__img_ocr, i, parallel_id, img, chars, self.parallel_limiter[parallel_id])
                            if use_delay:
                                await trio.sleep(0.1)  # 小文档保留延迟，避免任务调度过于密集
                    concurrent_elapsed = timer() - concurrent_start
                    print(f"[CONCURRENCY DEBUG] All concurrent tasks completed in {concurrent_elapsed:.3f}s")
            else:
                # 串行模式（虽然串行，但仍使用线程池避免阻塞事件循环）
                print(f"[CONCURRENCY DEBUG] Using SEQUENTIAL mode")
                sequential_start = timer()
                print(f"[CONCURRENCY DEBUG] Sequential start time: {sequential_start:.3f}")
                for i, img in enumerate(self.page_images):
                    chars = await trio.to_thread.run_sync(__ocr_preprocess, i, img)
                    print(f"[CONCURRENCY DEBUG] Processing page {i+1}/{total_pages} sequentially at {timer():.3f}")
                    await __img_ocr(i, 0, img, chars, None)
                sequential_elapsed = timer() - sequential_start
                print(f"[CONCURRENCY DEBUG] All sequential tasks completed in {sequential_elapsed:.3f}s")

        start = timer()

        trio.run(__img_ocr_launcher)

        logging.info(f"__images__ {len(self.page_images)} pages cost {timer() - start}s")

        if not self.is_english and not any([c for c in self.page_chars]) and self.boxes:
            bxes = [b for bxs in self.boxes for b in bxs]
            self.is_english = re.search(r"[\na-zA-Z0-9,/¸;:'\[\]\(\)!@#$%^&*\"?<>._-]{30,}", "".join([b["text"] for b in random.choices(bxes, k=min(30, len(bxes)))]))

        logging.debug("Is it English:", self.is_english)

        self.page_cum_height = np.cumsum(self.page_cum_height)
        assert len(self.page_cum_height) == len(self.page_images) + 1
        if len(self.boxes) == 0 and zoomin < 9:
            self.__images__(fnm, zoomin * 3, page_from, page_to, callback)

    def __call__(self, fnm, need_image=True, zoomin=3, return_html=False):
        self.__images__(fnm, zoomin)
        self._layouts_rec(zoomin)
        self._table_transformer_job(zoomin)
        self._text_merge()
        self._concat_downward()
        self._filter_forpages()
        tbls = self._extract_table_figure(need_image, zoomin, return_html, False)
        return self.__filterout_scraps(deepcopy(self.boxes), zoomin), tbls

    @staticmethod
    def remove_tag(txt):
        return re.sub(r"@@[\t0-9.-]+?##", "", txt)

    def crop(self, text, ZM=3, need_position=False):
        imgs = []
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", text):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        if not poss:
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + GAP), min(self.page_images[pos[0][-1]].size[1] / ZM, pos[4] + 120)))

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width
            bottom *= ZM
            for pn in pns[1:]:
                bottom += self.page_images[pn - 1].size[1]
            imgs.append(self.page_images[pns[0]].crop((left * ZM, top * ZM, right * ZM, min(bottom, self.page_images[pns[0]].size[1]))))
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, left, right, top, min(bottom, self.page_images[pns[0]].size[1]) / ZM))
            bottom -= self.page_images[pns[0]].size[1]
            for pn in pns[1:]:
                imgs.append(self.page_images[pn].crop((left * ZM, 0, right * ZM, min(bottom, self.page_images[pn].size[1]))))
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, left, right, 0, min(bottom, self.page_images[pn].size[1]) / ZM))
                bottom -= self.page_images[pn].size[1]

        if not imgs:
            if need_position:
                return None, None
            return
        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    def get_position(self, bx, ZM):
        poss = []
        pn = bx["page_number"]
        top = bx["top"] - self.page_cum_height[pn - 1]
        bott = bx["bottom"] - self.page_cum_height[pn - 1]
        poss.append((pn, bx["x0"], bx["x1"], top, min(bott, self.page_images[pn - 1].size[1] / ZM)))
        while bott * ZM > self.page_images[pn - 1].size[1]:
            bott -= self.page_images[pn - 1].size[1] / ZM
            top = 0
            pn += 1
            poss.append((pn, bx["x0"], bx["x1"], top, min(bott, self.page_images[pn - 1].size[1] / ZM)))
        return poss


class PlainParser:
    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        self.outlines = []
        lines = []
        try:
            self.pdf = pdf2_read(filename if isinstance(filename, str) else BytesIO(filename))
            for page in self.pdf.pages[from_page:to_page]:
                lines.extend([t for t in page.extract_text().split("\n")])

            outlines = self.pdf.outline

            def dfs(arr, depth):
                for a in arr:
                    if isinstance(a, dict):
                        self.outlines.append((a["/Title"], depth))
                        continue
                    dfs(a, depth + 1)

            dfs(outlines, 0)
        except Exception:
            logging.exception("Outlines exception")
        if not self.outlines:
            logging.warning("Miss outlines")

        return [(line, "") for line in lines], []

    def crop(self, ck, need_position):
        raise NotImplementedError

    @staticmethod
    def remove_tag(txt):
        raise NotImplementedError


class VisionParser(RAGFlowPdfParser):
    def __init__(self, vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model

    def __images__(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        try:
            with sys.modules[LOCK_KEY_pdfplumber]:
                self.pdf = pdfplumber.open(fnm) if isinstance(fnm, str) else pdfplumber.open(BytesIO(fnm))
                self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]
                self.total_page = len(self.pdf.pages)
        except Exception:
            self.page_images = None
            self.total_page = 0
            logging.exception("VisionParser __images__")

    def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
        callback = kwargs.get("callback", lambda prog, msg: None)

        self.__images__(fnm=filename, zoomin=3, page_from=from_page, page_to=to_page, **kwargs)

        total_pdf_pages = self.total_page

        start_page = max(0, from_page)
        end_page = min(to_page, total_pdf_pages)

        all_docs = []

        for idx, img_binary in enumerate(self.page_images or []):
            pdf_page_num = idx  # 0-based
            if pdf_page_num < start_page or pdf_page_num >= end_page:
                continue

            docs = picture_vision_llm_chunk(
                binary=img_binary,
                vision_model=self.vision_model,
                prompt=vision_llm_describe_prompt(page=pdf_page_num + 1),
                callback=callback,
            )

            if docs:
                all_docs.append(docs)
        return [(doc, "") for doc in all_docs], []


if __name__ == "__main__":
    pass
