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
import copy
import time
import os

from huggingface_hub import snapshot_download

from ..depend.file_utils import get_project_base_directory
from ..depend.settings import PARALLEL_DEVICES
from .operators import *  # noqa: F403
from . import operators
import math
import numpy as np
import cv2
import onnxruntime as ort

from .postprocess import build_post_process

loaded_models = {}

# Global OCR instance cache for model preloading
_ocr_instance_cache = {}
_ocr_cache_lock = None

def _get_ocr_cache_lock():
    """Get or create a lock for OCR cache (thread-safe initialization)"""
    global _ocr_cache_lock
    if _ocr_cache_lock is None:
        import threading
        _ocr_cache_lock = threading.Lock()
    return _ocr_cache_lock


def get_or_create_ocr_instance(model_dir=None, use_cache=True):
    """
    Get or create OCR instance with optional caching.
    
    Args:
        model_dir: Model directory path (optional)
        use_cache: Whether to use cached instance (default: True)
                   Set to False to force creating a new instance
    
    Returns:
        OCR instance
    """
    # Check if caching is enabled via environment variable
    if not use_cache:
        enable_cache = os.environ.get("DEEPDOC_OCR_CACHE_ENABLED", "1").lower() in ("1", "true", "yes")
        use_cache = enable_cache
    
    if not use_cache:
        # Create new instance without caching
        return OCR(model_dir)
    
    # Generate cache key based on configuration
    gpu_sessions = int(os.environ.get("DEEPDOC_GPU_SESSIONS", "0"))
    parallel_devices = PARALLEL_DEVICES
    
    # Create cache key: model_dir + gpu_sessions + parallel_devices
    cache_key = f"{model_dir or 'default'}_{gpu_sessions}_{parallel_devices}"
    
    lock = _get_ocr_cache_lock()
    with lock:
        if cache_key not in _ocr_instance_cache:
            print(f"[OCR CACHE] Creating new OCR instance (cache_key={cache_key})")
            _ocr_instance_cache[cache_key] = OCR(model_dir)
        else:
            print(f"[OCR CACHE] Reusing cached OCR instance (cache_key={cache_key})")
        return _ocr_instance_cache[cache_key]


def preload_ocr_model(model_dir=None):
    """
    Preload OCR model to cache (useful for reducing startup latency).
    
    Args:
        model_dir: Model directory path (optional)
    
    Returns:
        OCR instance
    """
    print(f"[OCR PRELOAD] Preloading OCR model...")
    start_time = time.time()
    ocr = get_or_create_ocr_instance(model_dir, use_cache=True)
    elapsed = time.time() - start_time
    print(f"[OCR PRELOAD] OCR model preloaded in {elapsed:.2f} seconds")
    return ocr


def clear_ocr_cache():
    """Clear OCR instance cache (useful for memory management or testing)"""
    lock = _get_ocr_cache_lock()
    with lock:
        _ocr_instance_cache.clear()
        print("[OCR CACHE] OCR instance cache cleared")

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(
        op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops


def load_model(model_dir, nm, device_id: int | None = None, session_id: int = 0):
    """
    Load ONNX model with support for multi-session on single GPU.
    
    Args:
        model_dir: Directory containing the model file
        nm: Model name (without .onnx extension)
        device_id: GPU device ID (None for CPU, 0 for first GPU, etc.)
        session_id: Session ID for multi-session support on same GPU (default: 0)
    """
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    # Include both device_id and session_id in cache key to support multi-session
    if device_id is not None:
        model_cached_tag = f"{model_file_path}_{device_id}_{session_id}"
    else:
        model_cached_tag = f"{model_file_path}_cpu_{session_id}"

    global loaded_models
    loaded_model = loaded_models.get(model_cached_tag)
    if loaded_model:
        logging.info(f"load_model {model_file_path} reuses cached model (device_id={device_id}, session_id={session_id})")
        return loaded_model

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))

    def cuda_is_available():
        """
        Check if CUDA is available for ONNX Runtime.
        Returns True only if:
        1. ONNX Runtime has CUDAExecutionProvider available
        2. PyTorch CUDA is available (optional, for device count check)
        3. Device ID is valid
        """
        try:
            # First check if ONNX Runtime supports CUDA
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in available_providers:
                logging.warning(f"CUDAExecutionProvider not available in ONNX Runtime. Available providers: {available_providers}")
                return False
            
            # Then check PyTorch CUDA (if available) for device count
            try:
                import torch
                if device_id is not None:
                    if not torch.cuda.is_available():
                        logging.warning("PyTorch CUDA is not available")
                        return False
                    if torch.cuda.device_count() <= device_id:
                        logging.warning(f"Device {device_id} not available. Only {torch.cuda.device_count()} GPU(s) available")
                        return False
            except ImportError:
                # PyTorch not installed, but ONNX Runtime has CUDA support
                # Assume device_id 0 is available if not specified
                if device_id is not None and device_id > 0:
                    logging.warning(f"PyTorch not available, cannot verify device {device_id}. Assuming device 0 is available.")
                    return device_id == 0
            
            return True
        except Exception as e:
            logging.warning(f"Error checking CUDA availability: {e}")
            return False

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2

    # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    
    # Get GPU memory limit per session (default: 4GB, can be overridden)
    gpu_mem_limit_mb = int(os.environ.get("DEEPDOC_GPU_MEM_LIMIT_MB", "4096"))
    gpu_mem_limit = gpu_mem_limit_mb * 1024 * 1024
    
    if cuda_is_available():
        # ONNX Runtime provider_options values must be strings for some versions
        # Convert all values to strings to ensure compatibility
        cuda_provider_options = {
            "device_id": str(int(device_id)) if device_id is not None else "0",  # Ensure integer then string
            "gpu_mem_limit": str(gpu_mem_limit),  # Limit gpu memory per session
            "arena_extend_strategy": "kNextPowerOfTwo",  # gpu memory allocation strategy
        }
        try:
            sess = ort.InferenceSession(
                model_file_path,
                options=options,
                providers=['CUDAExecutionProvider'],
                provider_options=[cuda_provider_options]
            )
        except Exception as e:
            # Fallback: try without provider_options (use default CUDA device)
            logging.warning(f"Failed to create CUDA session with options: {e}. Trying without options...")
            try:
                sess = ort.InferenceSession(
                    model_file_path,
                    options=options,
                    providers=['CUDAExecutionProvider']
                )
            except Exception as e2:
                logging.warning(f"Failed to create CUDA session: {e2}. Falling back to CPU.")
                sess = ort.InferenceSession(
                    model_file_path,
                    options=options,
                    providers=['CPUExecutionProvider']
                )
                run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
                logging.info(f"load_model {model_file_path} uses CPU (fallback, session_id={session_id})")
                loaded_model = (sess, run_options)
                loaded_models[model_cached_tag] = loaded_model
                return loaded_model
        # Note: memory.enable_memory_arena_shrinkage should be set only once per device
        # For multi-session on same GPU, only set it for the first session (session_id=0)
        # This avoids the "Did not find an arena based allocator" error
        actual_device_id = device_id if device_id is not None else 0
        if session_id == 0:
            try:
                run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:" + str(actual_device_id))
                logging.info(f"load_model {model_file_path} set memory arena shrinkage for device {actual_device_id} (first session)")
            except Exception as e:
                # If the config entry already exists or fails, log and continue
                logging.warning(f"Failed to set memory arena shrinkage for device {actual_device_id}, session {session_id}: {e}")
        logging.info(f"load_model {model_file_path} uses GPU (device_id={actual_device_id}, session_id={session_id}, mem_limit={gpu_mem_limit_mb}MB)")
    else:
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CPUExecutionProvider'])
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        logging.info(f"load_model {model_file_path} uses CPU (session_id={session_id})")
    loaded_model = (sess, run_options)
    loaded_models[model_cached_tag] = loaded_model
    return loaded_model


class TextRecognizer:
    def __init__(self, model_dir, device_id: int | None = None, session_id: int = 0):
        self.rec_image_shape = [int(v) for v in "3, 48, 320".split(",")]
        # 支持通过环境变量 DEEPDOC_REC_BATCH_NUM 配置批处理大小
        # 默认使用 8（测试验证的最优配置：368秒，0.78页/秒）
        self.rec_batch_num = int(os.environ.get("DEEPDOC_REC_BATCH_NUM", "8"))
        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": os.path.join(model_dir, "ocr.res"),
            "use_space_char": True
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, 'rec', device_id, session_id)
        self.input_tensor = self.predictor.get_inputs()[0]

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        w = self.input_tensor.shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_vl(self, img, image_shape):

        imgC, imgH, imgW = image_shape
        img = img[:, :, ::-1]  # bgr2rgb
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        return resized_image

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0:img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            self.srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

    def resize_norm_img_sar(self, img, image_shape,
                            width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h = img.shape[0]
        w = img.shape[1]
        valid_ratio = 1.0
        # make sure new_width is an integral multiple of width_divisor.
        width_divisor = int(1 / width_downsample_ratio)
        # resize
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        if imgW_min is not None:
            resize_w = max(imgW_min, resize_w)
        if imgW_max is not None:
            valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
            resize_w = min(imgW_max, resize_w)
        resized_image = cv2.resize(img, (resize_w, imgH))
        resized_image = resized_image.astype('float32')
        # norm
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        resize_shape = resized_image.shape
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        pad_shape = padding_im.shape

        return padding_im, resize_shape, pad_shape, valid_ratio

    def resize_norm_img_spin(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # return padding_im
        img = cv2.resize(img, tuple([100, 32]), cv2.INTER_CUBIC)
        img = np.array(img, np.float32)
        img = np.expand_dims(img, -1)
        img = img.transpose((2, 0, 1))
        mean = [127.5]
        std = [127.5]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        mean = np.float32(mean.reshape(1, -1))
        stdinv = 1 / np.float32(std.reshape(1, -1))
        img -= mean
        img *= stdinv
        return img

    def resize_norm_img_svtr(self, img, image_shape):

        imgC, imgH, imgW = image_shape
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image

    def resize_norm_img_abinet(self, img, image_shape):

        imgC, imgH, imgW = image_shape

        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        resized_image = (
            resized_image - mean[None, None, ...]) / std[None, None, ...]
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype('float32')

        return resized_image

    def norm_img_can(self, img, image_shape):

        img = cv2.cvtColor(
            img, cv2.COLOR_BGR2GRAY)  # CAN only predict gray scale image

        if self.rec_image_shape[0] == 1:
            h, w = img.shape
            _, imgH, imgW = self.rec_image_shape
            if h < imgH or w < imgW:
                padding_h = max(imgH - h, 0)
                padding_w = max(imgW - w, 0)
                img_padded = np.pad(img, ((0, padding_h), (0, padding_w)),
                                    'constant',
                                    constant_values=(255))
                img = img_padded

        img = np.expand_dims(img, 0) / 255.0  # h,w,c -> c,h,w
        img = img.astype('float32')

        return img

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            for i in range(100000):
                try:
                    outputs = self.predictor.run(None, input_dict, self.run_options)
                    break
                except Exception as e:
                    if i >= 3:
                        raise e
                    time.sleep(5)
            preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res, time.time() - st


class TextDetector:
    def __init__(self, model_dir, device_id: int | None = None, session_id: int = 0):
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': "max",
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {"name": "DBPostProcess", "thresh": 0.3, "box_thresh": 0.5, "max_candidates": 1000,
                              "unclip_ratio": 1.5, "use_dilation": False, "score_mode": "fast", "box_type": "quad"}

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, 'det', device_id, session_id)
        self.input_tensor = self.predictor.get_inputs()[0]

        img_h, img_w = self.input_tensor.shape[2:]
        if isinstance(img_h, str) or isinstance(img_w, str):
            pass
        elif img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'image_shape': [img_h, img_w]
                }
            }
        self.preprocess_op = create_operators(pre_process_list)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        input_dict = {}
        input_dict[self.input_tensor.name] = img
        for i in range(100000):
            try:
                outputs = self.predictor.run(None, input_dict, self.run_options)
                break
            except Exception as e:
                if i >= 3:
                    raise e
                time.sleep(5)

        post_result = self.postprocess_op({"maps": outputs[0]}, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes, time.time() - st


class OCR:
    def __init__(self, model_dir=None):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        _ocr_init_start = time.perf_counter()
        print(f"[TIMING] OCR.__init__ 开始: {_ocr_init_start:.6f}")
        
        if not model_dir:
            try:
                model_dir = os.path.join(
                        get_project_base_directory(),
                        "dict/ocr")

                # 在 model_dir 设置后添加
                print("模型目录：", model_dir)
                det_model_path = os.path.join(model_dir, "det.onnx")
                print("det.onnx 是否存在：", os.path.exists(det_model_path))

                # Support multi-GPU or multi-session on single GPU
                # Check if multi-session mode is enabled (for single GPU)
                gpu_sessions = int(os.environ.get("DEEPDOC_GPU_SESSIONS", "0"))
                
                if PARALLEL_DEVICES > 0:
                    if gpu_sessions > 0:
                        # Multi-session mode: create multiple sessions on single GPU (device_id=0)
                        print(f"[OCR INIT] Using multi-session mode: {gpu_sessions} sessions on GPU 0")
                        self.text_detector = []
                        self.text_recognizer = []
                        for session_id in range(gpu_sessions):
                            self.text_detector.append(TextDetector(model_dir, device_id=0, session_id=session_id))
                            self.text_recognizer.append(TextRecognizer(model_dir, device_id=0, session_id=session_id))
                    else:
                        # Multi-GPU mode: create one session per GPU
                        self.text_detector = []
                        self.text_recognizer = []
                        for device_id in range(PARALLEL_DEVICES):
                            self.text_detector.append(TextDetector(model_dir, device_id))
                            self.text_recognizer.append(TextRecognizer(model_dir, device_id))
                else:
                    self.text_detector = [TextDetector(model_dir)]
                    self.text_recognizer = [TextRecognizer(model_dir)]

            except Exception:
                model_dir = snapshot_download(repo_id="InfiniFlow/deepdoc",
                                              local_dir=os.path.join(get_project_base_directory(), "dict/ocr"),
                                              local_dir_use_symlinks=False)
                
                # Support multi-GPU or multi-session on single GPU
                gpu_sessions = int(os.environ.get("DEEPDOC_GPU_SESSIONS", "0"))
                
                if PARALLEL_DEVICES > 0:
                    if gpu_sessions > 0:
                        # Multi-session mode: create multiple sessions on single GPU (device_id=0)
                        print(f"[OCR INIT] Using multi-session mode: {gpu_sessions} sessions on GPU 0")
                        self.text_detector = []
                        self.text_recognizer = []
                        for session_id in range(gpu_sessions):
                            self.text_detector.append(TextDetector(model_dir, device_id=0, session_id=session_id))
                            self.text_recognizer.append(TextRecognizer(model_dir, device_id=0, session_id=session_id))
                    else:
                        # Multi-GPU mode: create one session per GPU
                        self.text_detector = []
                        self.text_recognizer = []
                        for device_id in range(PARALLEL_DEVICES):
                            self.text_detector.append(TextDetector(model_dir, device_id))
                            self.text_recognizer.append(TextRecognizer(model_dir, device_id))
                else:
                    self.text_detector = [TextDetector(model_dir)]
                    self.text_recognizer = [TextRecognizer(model_dir)]

        self.drop_score = 0.5
        self.crop_image_res_index = 0
        
        _ocr_init_end = time.perf_counter()
        _ocr_init_duration = _ocr_init_end - _ocr_init_start
        print(f"[TIMING] OCR.__init__ 完成: {_ocr_init_end:.6f} (耗时: {_ocr_init_duration:.3f} 秒)")

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            # Try original orientation
            rec_result = self.text_recognizer[0]([dst_img])
            text, score = rec_result[0][0]
            best_score = score
            best_img = dst_img

            # Try clockwise 90° rotation
            rotated_cw = np.rot90(dst_img, k=3)
            rec_result = self.text_recognizer[0]([rotated_cw])
            rotated_cw_text, rotated_cw_score = rec_result[0][0]
            if rotated_cw_score > best_score:
                best_score = rotated_cw_score
                best_img = rotated_cw

            # Try counter-clockwise 90° rotation
            rotated_ccw = np.rot90(dst_img, k=1)
            rec_result = self.text_recognizer[0]([rotated_ccw])
            rotated_ccw_text, rotated_ccw_score = rec_result[0][0]
            if rotated_ccw_score > best_score:
                best_img = rotated_ccw

            # Use the best image
            dst_img = best_img
        return dst_img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def detect(self, img, device_id: int | None = None):
        # 如果 device_id 为 None，使用第一个 detector（索引 0）
        # 在 CPU 模式下，text_detector 只有一个元素，所以使用 0
        # 在 GPU 模式下，如果 device_id 为 None，也使用 0（第一个 GPU）
        if device_id is None:
            device_id = 0
        
        # 确保 device_id 在有效范围内
        if device_id >= len(self.text_detector):
            device_id = 0

        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            return None, None, time_dict

        start = time.time()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        return zip(self.sorted_boxes(dt_boxes), [
                   ("", 0) for _ in range(len(dt_boxes))])

    def recognize(self, ori_im, box, device_id: int | None = None):
        if device_id is None:
            device_id = 0
        
        # 确保 device_id 在有效范围内
        if device_id >= len(self.text_recognizer):
            device_id = 0

        img_crop = self.get_rotate_crop_image(ori_im, box)

        rec_res, elapse = self.text_recognizer[device_id]([img_crop])
        text, score = rec_res[0]
        if score < self.drop_score:
            return ""
        return text

    def recognize_batch(self, img_list, device_id: int | None = None):
        if device_id is None:
            device_id = 0
        
        # 确保 device_id 在有效范围内
        if device_id >= len(self.text_recognizer):
            device_id = 0
        
        rec_res, elapse = self.text_recognizer[device_id](img_list)
        texts = []
        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if score < self.drop_score:
                text = ""
            texts.append(text)
        return texts

    def __call__(self, img, device_id = 0, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        if device_id is None:
            device_id = 0

        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = self.text_recognizer[device_id](img_crop_list)

        time_dict['rec'] = elapse

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start

        # for bno in range(len(img_crop_list)):
        #    print(f"{bno}, {rec_res[bno]}")

        return list(zip([a.tolist() for a in filter_boxes], filter_rec_res))
