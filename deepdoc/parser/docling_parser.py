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
from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Iterable, Optional
from urllib.parse import unquote

import pdfplumber
from PIL import Image

try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None  

try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
except Exception:
    class RAGFlowPdfParser:  
        pass


class DoclingContentType(str, Enum):
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    EQUATION = "equation"


@dataclass
class _BBox:
    page_no: int  
    x0: float
    y0: float
    x1: float
    y1: float


class DoclingParser(RAGFlowPdfParser):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.page_images: list[Image.Image] = []
        self.page_from = 0
        self.page_to = 10_000
        self.outlines = []
   
        
    def check_installation(self) -> bool:
        if DocumentConverter is None:
            self.logger.warning("[Docling] 'docling' is not importable, please: pip install docling")
            return False
        try:
            _ = DocumentConverter()
            return True
        except Exception as e:
            self.logger.error(f"[Docling] init DocumentConverter failed: {e}")
            return False

    def __images__(self, fnm, zoomin: int = 1, page_from=0, page_to=600, callback=None):
        self.page_from = page_from
        self.page_to = page_to
        bytes_io = None
        try:
            if not isinstance(fnm, (str, PathLike)):
                bytes_io = BytesIO(fnm)

            opener = pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(bytes_io)
            with opener as pdf:
                pages = pdf.pages[page_from:page_to]
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for p in pages]
        except Exception as e:
            self.page_images = []
            self.logger.exception(e)
        finally:
            if bytes_io:
                bytes_io.close()

    def _make_line_tag(self,bbox: _BBox) -> str:
        if bbox is None:
            return ""
        x0,x1, top, bott = bbox.x0, bbox.x1, bbox.y0, bbox.y1
        if hasattr(self, "page_images") and self.page_images and len(self.page_images) >= bbox.page_no:
            _, page_height = self.page_images[bbox.page_no-1].size
            top, bott = page_height-top ,page_height-bott
        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format(
            bbox.page_no, x0,x1, top, bott
        )

    @staticmethod
    def extract_positions(txt: str) -> list[tuple[list[int], float, float, float, float]]:
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def crop(self, text: str, ZM: int = 1, need_position: bool = False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            return (None, None) if need_position else None

        GAP = 6
        pos = poss[0]
        poss.insert(0, ([pos[0][0]], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        poss.append(([pos[0][-1]], pos[1], pos[2], min(self.page_images[pos[0][-1]].size[1], pos[4] + GAP), min(self.page_images[pos[0][-1]].size[1], pos[4] + 120)))
        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            if bottom <= top:
                bottom = top + 4
            img0 = self.page_images[pns[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))
            
            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss)-1:
                positions.append((pns[0] + self.page_from, x0, x1, y0, y1))
            remain_bottom = bottom - img0.size[1]
            for pn in pns[1:]:
                if remain_bottom <= 0:
                    break
                page = self.page_images[pn]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(remain_bottom, page.size[1]))
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, x0, x1, y0, y1))
                remain_bottom -= page.size[1]

        if not imgs:
            return (None, None) if need_position else None

        height = sum(i.size[1] + GAP for i in imgs)
        width = max(i.size[0] for i in imgs)
        pic = Image.new("RGB", (width, int(height)), (245, 245, 245))
        h = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(h)))
            h += img.size[1] + GAP

        return (pic, positions) if need_position else pic

    def _iter_doc_items(self, doc, has_bbox: bool = True) -> Iterable[tuple[str, Any, Optional[_BBox], str]]:
        """
        Iterate over document items (texts, equations).
        
        Args:
            doc: Docling document object
            has_bbox: Whether the document format supports bbox (PDF=True, DOCX/PPTX=False)
            
        Yields:
            Tuple of (content_type, text, bbox, label) where:
            - content_type: DoclingContentType value
            - text: Text content
            - bbox: Bounding box (None for DOCX/PPTX)
            - label: Docling label (e.g., "section_header", "text", "list_item", "FORMULA")
        """
        for t in getattr(doc, "texts", []):
            parent = getattr(t, "parent", "")
            ref = getattr(parent, "cref", "") if parent else ""
            label = getattr(t, "label", "")
            # Accept section_header, text, and list_item labels
            # For DOCX/PPTX, ref may not be exactly "#/body" (could be "#/groups/0", "#/texts/0", etc.)
            # So we accept any ref for these labels, or specifically check for "#/body" when needed
            if label in ("section_header", "text", "list_item"):
                text = getattr(t, "text", "") or ""
                if not text.strip():
                    continue
                bbox = None
                if has_bbox and getattr(t, "prov", None):
                    pn = getattr(t.prov[0], "page_no", None)
                    bb = getattr(t.prov[0], "bbox", None)
                    if bb:
                        bb = [getattr(bb, "l", None), getattr(bb, "t", None), getattr(bb, "r", None), getattr(bb, "b", None)]
                        if pn and bb and len(bb) == 4 and all(b is not None for b in bb):
                            bbox = _BBox(page_no=int(pn), x0=bb[0], y0=bb[1], x1=bb[2], y1=bb[3])
                yield (DoclingContentType.TEXT.value, text, bbox, label)

        for item in getattr(doc, "texts", []):
            item_label = getattr(item, "label", "")
            if item_label in ("FORMULA",):
                text = getattr(item, "text", "") or ""
                bbox = None
                if has_bbox and getattr(item, "prov", None):
                    pn = getattr(item.prov, "page_no", None)
                    bb = getattr(item.prov, "bbox", None)
                    if bb:
                        bb = [getattr(bb, "l", None), getattr(bb, "t", None), getattr(bb, "r", None), getattr(bb, "b", None)]
                        if pn and bb and len(bb) == 4 and all(b is not None for b in bb):
                            bbox = _BBox(int(pn), bb[0], bb[1], bb[2], bb[3])
                yield (DoclingContentType.EQUATION.value, text, bbox, item_label)

    def _label_to_style(self, label: str) -> str:
        """
        Map Docling label to Word style name.
        
        Args:
            label: Docling label (e.g., "section_header", "text", "list_item")
            
        Returns:
            Word-style name (e.g., "Heading", "Normal", "List Item")
        """
        label_to_style_map = {
            "section_header": "Heading",
            "text": "Normal",
            "list_item": "List Item",
            "FORMULA": "Equation",
        }
        return label_to_style_map.get(label, "Normal")

    def _transfer_to_sections(self, doc, parse_method: str, has_bbox: bool = True) -> list[tuple[str, str]]:
        """
        Transfer document items to sections.
        
        Args:
            doc: Docling document object
            parse_method: Parsing method ("raw", "manual", "paper")
            has_bbox: Whether the document format supports bbox
            
        Returns:
            List of (text, tag_or_style) tuples where:
            - For PDF (has_bbox=True): tag is position tag (e.g., "@@1\t0.0\t100.0\t0.0\t50.0##")
            - For DOCX/PPTX (has_bbox=False): tag is style name (e.g., "Heading", "Normal")
        """
        sections: list[tuple[str, str]] = []
        for typ, payload, bbox, label in self._iter_doc_items(doc, has_bbox=has_bbox):
            if typ == DoclingContentType.TEXT.value:
                section = payload.strip()
                if not section:
                    continue
            elif typ == DoclingContentType.EQUATION.value:
                section = payload.strip()
            else:
                continue
            
            # For PDF (has_bbox=True): use position tag
            # For DOCX/PPTX (has_bbox=False): use label as style
            if isinstance(bbox, _BBox):
                tag = self._make_line_tag(bbox)
            else:
                # No bbox, use label as style for DOCX/PPTX
                tag = self._label_to_style(label)
            
            if parse_method == "manual":
                sections.append((section, typ, tag))
            elif parse_method == "paper":
                sections.append((section + tag, typ))
            else:
                sections.append((section, tag))
        return sections

    def cropout_docling_table(self, page_no: int, bbox: tuple[float, float, float, float], zoomin: int = 1):
        if not getattr(self, "page_images", None):
            return None, ""

        idx = (page_no - 1) - getattr(self, "page_from", 0)
        if idx < 0 or idx >= len(self.page_images):
            return None, ""

        page_img = self.page_images[idx]
        W, H = page_img.size
        left, top, right, bott = bbox

        x0 = float(left)
        y0 = float(H-top)
        x1 = float(right)
        y1 = float(H-bott)

        x0, y0 = max(0.0, min(x0, W - 1)), max(0.0, min(y0, H - 1))
        x1, y1 = max(x0 + 1.0, min(x1, W)), max(y0 + 1.0, min(y1, H))

        try:
            crop = page_img.crop((int(x0), int(y0), int(x1), int(y1))).convert("RGB")
        except Exception:
            return None, ""

        pos = (page_no-1 if page_no>0 else 0, x0, x1, y0, y1)
        return crop, [pos]

    def _transfer_to_tables(self, doc, has_bbox: bool = True):
        """
        Transfer document tables and pictures to tables format.
        
        Args:
            doc: Docling document object
            has_bbox: Whether the document format supports bbox
        """
        tables = []
        for tab in getattr(doc, "tables", []):
            img = None
            positions = ""
            if has_bbox and getattr(tab, "prov", None):
                pn = getattr(tab.prov[0], "page_no", None)
                bb = getattr(tab.prov[0], "bbox", None)
                if pn is not None and bb is not None:
                    left = getattr(bb, "l", None)
                    top = getattr(bb, "t", None)
                    right = getattr(bb, "r", None)
                    bott = getattr(bb, "b", None)
                    if None not in (left, top, right, bott):
                        img, positions = self.cropout_docling_table(int(pn), (float(left), float(top), float(right), float(bott)))
            html = ""
            try:
                html = tab.export_to_html(doc=doc)
            except Exception:
                pass
            tables.append(((img, html), positions if positions else ""))
        
        # Handle pictures (for PDF with bbox)
        if has_bbox:
            for pic in getattr(doc, "pictures", []):
                img = None
                positions = ""
                if getattr(pic, "prov", None):
                    pn = getattr(pic.prov[0], "page_no", None)
                    bb = getattr(pic.prov[0], "bbox", None)
                    if pn is not None and bb is not None:
                        left = getattr(bb, "l", None)
                        top = getattr(bb, "t", None)
                        right = getattr(bb, "r", None)
                        bott = getattr(bb, "b", None)
                        if None not in (left, top, right, bott):
                            img, positions = self.cropout_docling_table(int(pn), (float(left), float(top), float(right), float(bott)))
                captions = ""
                try:
                    captions = pic.caption_text(doc=doc)
                except Exception:
                    pass
                tables.append(((img, [captions]), positions if positions else ""))
        
        return tables

    def _extract_image_from_data_uri(self, data_uri: str) -> Optional[Image.Image]:
        """
        Extract PIL Image from base64 data URI.
        
        Args:
            data_uri: Data URI string (e.g., "data:image/png;base64,...")
            
        Returns:
            PIL Image object or None if extraction fails
        """
        try:
            # Parse data URI: data:image/png;base64,<base64_data>
            if not data_uri.startswith("data:"):
                return None
            
            # Extract base64 part
            if "," in data_uri:
                base64_data = data_uri.split(",", 1)[1]
            else:
                return None
            
            # Decode base64
            image_data = base64.b64decode(base64_data)
            
            # Create PIL Image
            img = Image.open(BytesIO(image_data))
            return img.convert("RGB")
        except Exception as e:
            self.logger.warning(f"[Docling] Failed to extract image from data URI: {e}")
            return None

    def _find_element_caption(
        self, doc, element, element_type: str, element_idx: int, caption_keywords: list[str]
    ) -> str:
        """
        Find caption for an element (picture or table) by checking document structure.
        
        For DOCX, captions are text items that follow the element in the parent's children list.
        
        Args:
            doc: Docling document object
            element: The element object (picture or table)
            element_type: Type of element ("picture" or "table")
            element_idx: Index of the element in doc.pictures or doc.tables
            caption_keywords: List of keywords to identify captions (e.g., ["图表", "figure"] for pictures)
            
        Returns:
            Caption text or empty string
        """
        try:
            if not hasattr(element, "parent") or not element.parent:
                return ""
            
            parent_ref = str(element.parent.cref) if hasattr(element.parent, "cref") else ""
            if not parent_ref or not parent_ref.startswith("#/texts/"):
                return ""
            
            # Find parent text item
            parent_idx = int(parent_ref.split("/")[-1])
            if parent_idx >= len(doc.texts):
                return ""
            
            parent_text = doc.texts[parent_idx]
            if not hasattr(parent_text, "children") or not parent_text.children:
                return ""
            
            # Find element in children list
            element_ref = f"#/{element_type}s/{element_idx}"
            element_idx_in_children = None
            for idx, child in enumerate(parent_text.children):
                child_ref = str(child.cref) if hasattr(child, "cref") else ""
                if child_ref == element_ref:
                    element_idx_in_children = idx
                    break
            
            if element_idx_in_children is None:
                return ""
            
            # Check next item after element (potential caption)
            if element_idx_in_children + 1 < len(parent_text.children):
                next_child = parent_text.children[element_idx_in_children + 1]
                next_ref = str(next_child.cref) if hasattr(next_child, "cref") else ""
                
                if next_ref.startswith("#/texts/"):
                    text_idx = int(next_ref.split("/")[-1])
                    if text_idx < len(doc.texts):
                        caption_text = doc.texts[text_idx]
                        text = getattr(caption_text, "text", "") or getattr(caption_text, "orig", "")
                        # Check if it looks like a caption based on keywords
                        if text and any(keyword in text.lower() for keyword in caption_keywords):
                            return text.strip()
            
            return ""
        except Exception as e:
            self.logger.warning(f"[Docling] Failed to find {element_type} caption: {e}")
            return ""

    def _find_picture_caption(self, doc, picture_idx: int) -> str:
        """
        Find caption for a picture by checking document structure.
        
        For DOCX, captions are not directly in PictureItem.captions,
        but are text items that follow the picture in the parent's children list.
        
        Args:
            doc: Docling document object
            picture_idx: Index of the picture in doc.pictures
            
        Returns:
            Caption text or empty string
        """
        try:
            pic = doc.pictures[picture_idx]
            return self._find_element_caption(
                doc, pic, "picture", picture_idx, ["图表", "figure", "图", "fig"]
            )
        except Exception as e:
            self.logger.warning(f"[Docling] Failed to find picture caption: {e}")
            return ""

    def _find_table_caption(self, doc, table_idx: int) -> str:
        """
        Find caption for a table by checking document structure.
        
        For DOCX, captions are text items that follow the table in the document structure.
        Similar to picture captions, but we also check for "Table" keywords.
        
        Args:
            doc: Docling document object
            table_idx: Index of the table in doc.tables
            
        Returns:
            Caption text or empty string
        """
        try:
            tab = doc.tables[table_idx]
            return self._find_element_caption(
                doc, tab, "table", table_idx, ["表", "table", "表格"]
            )
        except Exception as e:
            self.logger.warning(f"[Docling] Failed to find table caption: {e}")
            return ""

    def _transfer_to_tables_docx(self, doc) -> list[tuple[tuple, str]]:
        """
        Transfer DOCX document tables and pictures to tables format.
        DOCX doesn't have bbox, so we handle pictures differently.
        
        Args:
            doc: Docling document object
            
        Returns:
            List of ((image, html_or_captions), positions) tuples
        """
        tables = []
        
        # Handle tables
        for idx, tab in enumerate(getattr(doc, "tables", [])):
            html = ""
            try:
                html = tab.export_to_html(doc=doc)
            except Exception:
                pass
            
            # Find table caption through document structure
            caption = self._find_table_caption(doc, idx)
            
            # Also try direct caption_text method (might work for some cases)
            if not caption:
                try:
                    caption = tab.caption_text(doc=doc) if hasattr(tab, "caption_text") else ""
                except Exception:
                    pass
            
            # DOCX tables don't have bbox, so no image or positions
            # Format: ((None, html_or_captions), positions)
            # For tables with caption, we store as dict: {"caption": caption, "html": html}
            # For tables without caption, we store as string: html
            if caption:
                # Store caption and html together in a dict format
                table_data = {"caption": caption, "html": html}
                tables.append(((None, table_data), ""))
            else:
                tables.append(((None, html), ""))
        
        # Handle pictures
        for idx, pic in enumerate(getattr(doc, "pictures", [])):
            img = None
            captions = ""
            
            # Extract image from data URI
            if hasattr(pic, "image") and pic.image:
                if hasattr(pic.image, "uri"):
                    data_uri = str(pic.image.uri)
                    img = self._extract_image_from_data_uri(data_uri)
            
            # Find caption through document structure
            caption = self._find_picture_caption(doc, idx)
            if caption:
                captions = caption
            
            # Also try direct caption_text method (might work for some cases)
            if not captions:
                try:
                    captions = pic.caption_text(doc=doc)
                except Exception:
                    pass
            
            # DOCX pictures don't have bbox positions
            tables.append(((img, [captions] if captions else []), ""))
        
        return tables

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None, 
        lang: Optional[str] = None,        
        method: str = "auto",             
        delete_output: bool = True,
        parse_method: str = "raw"     
    ):

        if not self.check_installation():
            raise RuntimeError("Docling not available, please install `docling`")

        if binary is not None:
            tmpdir = Path(output_dir) if output_dir else Path.cwd() / ".docling_tmp"
            tmpdir.mkdir(parents=True, exist_ok=True)
            name = Path(filepath).name or "input.pdf"
            tmp_pdf = tmpdir / name
            with open(tmp_pdf, "wb") as f:
                if isinstance(binary, (bytes, bytearray)):
                    f.write(binary)
                else:
                    f.write(binary.getbuffer())
            src_path = tmp_pdf
        else:
            src_path = Path(filepath)
            if not src_path.exists():
                raise FileNotFoundError(f"PDF not found: {src_path}")

        if callback:
            callback(0.1, f"[Docling] Converting: {src_path}")

        try:
            self.__images__(str(src_path), zoomin=1)
        except Exception as e:
            self.logger.warning(f"[Docling] render pages failed: {e}")

        conv = DocumentConverter()  
        conv_res = conv.convert(str(src_path))
        doc = conv_res.document
        if callback:
            callback(0.7, f"[Docling] Parsed doc: {getattr(doc, 'num_pages', 'n/a')} pages")

        sections = self._transfer_to_sections(doc, parse_method=parse_method, has_bbox=True)
        tables = self._transfer_to_tables(doc, has_bbox=True)

        if callback:
            callback(0.95, f"[Docling] Sections: {len(sections)}, Tables: {len(tables)}")

        if binary is not None and delete_output:
            try:
                Path(src_path).unlink(missing_ok=True)
            except Exception:
                pass

        if callback:
            callback(1.0, "[Docling] Done.")
        return sections, tables

    def parse_docx(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        method: str = "auto",
        delete_output: bool = True,
        parse_method: str = "raw"
    ):
        """
        Parse DOCX file using Docling.
        
        Args:
            filepath: Path to DOCX file
            binary: Optional binary content of the file
            callback: Optional progress callback function
            output_dir: Optional temporary output directory
            lang: Optional language hint (not used for DOCX)
            method: Parsing method (not used for DOCX)
            delete_output: Whether to delete temporary files
            parse_method: Output format ("raw", "manual", "paper")
            
        Returns:
            Tuple of (sections, tables) where:
            - sections: List of (text, tag) or (text, type, tag) tuples
            - tables: List of ((image, html_or_captions), positions) tuples
        """
        if not self.check_installation():
            raise RuntimeError("Docling not available, please install `docling`")

        if binary is not None:
            tmpdir = Path(output_dir) if output_dir else Path.cwd() / ".docling_tmp"
            tmpdir.mkdir(parents=True, exist_ok=True)
            name = Path(filepath).name if filepath else "input.docx"
            if not name.endswith(".docx"):
                name = name + ".docx"
            tmp_docx = tmpdir / name
            with open(tmp_docx, "wb") as f:
                if isinstance(binary, (bytes, bytearray)):
                    f.write(binary)
                else:
                    f.write(binary.getbuffer())
            src_path = tmp_docx
        else:
            src_path = Path(filepath)
            if not src_path.exists():
                raise FileNotFoundError(f"DOCX not found: {src_path}")

        if callback:
            callback(0.1, f"[Docling] Converting DOCX: {src_path}")

        try:
            conv = DocumentConverter()
            conv_res = conv.convert(str(src_path))
            doc = conv_res.document
        except Exception as e:
            self.logger.error(f"[Docling] Failed to convert DOCX: {e}")
            raise

        if callback:
            callback(0.5, f"[Docling] Parsed DOCX: {len(getattr(doc, 'texts', []))} text items")

        # DOCX doesn't have bbox, so use has_bbox=False
        sections = self._transfer_to_sections(doc, parse_method=parse_method, has_bbox=False)
        tables = self._transfer_to_tables_docx(doc)

        if callback:
            callback(0.9, f"[Docling] Sections: {len(sections)}, Tables: {len(tables)}")

        if binary is not None and delete_output:
            try:
                Path(src_path).unlink(missing_ok=True)
            except Exception:
                pass

        if callback:
            callback(1.0, "[Docling] Done.")
        return sections, tables

    def parse_pptx(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        method: str = "auto",
        delete_output: bool = True,
        parse_method: str = "raw"
    ):
        """
        Parse PPTX file using Docling (preliminary support).
        
        Args:
            filepath: Path to PPTX file
            binary: Optional binary content of the file
            callback: Optional progress callback function
            output_dir: Optional temporary output directory
            lang: Optional language hint (not used for PPTX)
            method: Parsing method (not used for PPTX)
            delete_output: Whether to delete temporary files
            parse_method: Output format ("raw", "manual", "paper")
            
        Returns:
            Tuple of (sections, tables) where:
            - sections: List of (text, tag) or (text, type, tag) tuples
            - tables: List of ((image, html_or_captions), positions) tuples
        """
        if not self.check_installation():
            raise RuntimeError("Docling not available, please install `docling`")

        if binary is not None:
            tmpdir = Path(output_dir) if output_dir else Path.cwd() / ".docling_tmp"
            tmpdir.mkdir(parents=True, exist_ok=True)
            name = Path(filepath).name if filepath else "input.pptx"
            if not name.endswith(".pptx"):
                name = name + ".pptx"
            tmp_pptx = tmpdir / name
            with open(tmp_pptx, "wb") as f:
                if isinstance(binary, (bytes, bytearray)):
                    f.write(binary)
                else:
                    f.write(binary.getbuffer())
            src_path = tmp_pptx
        else:
            src_path = Path(filepath)
            if not src_path.exists():
                raise FileNotFoundError(f"PPTX not found: {src_path}")

        if callback:
            callback(0.1, f"[Docling] Converting PPTX: {src_path}")

        try:
            conv = DocumentConverter()
            conv_res = conv.convert(str(src_path))
            doc = conv_res.document
        except Exception as e:
            self.logger.error(f"[Docling] Failed to convert PPTX: {e}")
            raise

        if callback:
            callback(0.5, f"[Docling] Parsed PPTX: {len(getattr(doc, 'texts', []))} text items")

        # PPTX doesn't have bbox like DOCX
        sections = self._transfer_to_sections(doc, parse_method=parse_method, has_bbox=False)
        tables = self._transfer_to_tables_docx(doc)

        if callback:
            callback(0.9, f"[Docling] Sections: {len(sections)}, Tables: {len(tables)}")

        if binary is not None and delete_output:
            try:
                Path(src_path).unlink(missing_ok=True)
            except Exception:
                pass

        if callback:
            callback(1.0, "[Docling] Done.")
        return sections, tables

    def parse_xlsx(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes | None = None,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        method: str = "auto",
        delete_output: bool = True,
        parse_method: str = "raw"
    ):
        """
        Parse XLSX file using Docling (preliminary support).
        
        Args:
            filepath: Path to XLSX file
            binary: Optional binary content of the file
            callback: Optional progress callback function
            output_dir: Optional temporary output directory
            lang: Optional language hint (not used for XLSX)
            method: Parsing method (not used for XLSX)
            delete_output: Whether to delete temporary files
            parse_method: Output format ("raw", "manual", "paper")
            
        Returns:
            Tuple of (sections, tables) where:
            - sections: List of (text, tag) or (text, type, tag) tuples (usually empty for XLSX)
            - tables: List of ((image, html), positions) tuples
        """
        if not self.check_installation():
            raise RuntimeError("Docling not available, please install `docling`")

        if binary is not None:
            tmpdir = Path(output_dir) if output_dir else Path.cwd() / ".docling_tmp"
            tmpdir.mkdir(parents=True, exist_ok=True)
            name = Path(filepath).name if filepath else "input.xlsx"
            if not name.endswith(".xlsx"):
                name = name + ".xlsx"
            tmp_xlsx = tmpdir / name
            with open(tmp_xlsx, "wb") as f:
                if isinstance(binary, (bytes, bytearray)):
                    f.write(binary)
                else:
                    f.write(binary.getbuffer())
            src_path = tmp_xlsx
        else:
            src_path = Path(filepath)
            if not src_path.exists():
                raise FileNotFoundError(f"XLSX not found: {src_path}")

        if callback:
            callback(0.1, f"[Docling] Converting XLSX: {src_path}")

        try:
            conv = DocumentConverter()
            conv_res = conv.convert(str(src_path))
            doc = conv_res.document
        except Exception as e:
            self.logger.error(f"[Docling] Failed to convert XLSX: {e}")
            raise

        if callback:
            callback(0.5, f"[Docling] Parsed XLSX: {len(getattr(doc, 'tables', []))} tables")

        # XLSX is primarily tables, minimal text sections
        sections = self._transfer_to_sections(doc, parse_method=parse_method, has_bbox=False)
        # Use DOCX table handler (no bbox, similar structure)
        tables = self._transfer_to_tables_docx(doc)

        if callback:
            callback(0.9, f"[Docling] Sections: {len(sections)}, Tables: {len(tables)}")

        if binary is not None and delete_output:
            try:
                Path(src_path).unlink(missing_ok=True)
            except Exception:
                pass

        if callback:
            callback(1.0, "[Docling] Done.")
        return sections, tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = DoclingParser()
    print("Docling available:", parser.check_installation())
    sections, tables = parser.parse_pdf(filepath="test_docling/toc.pdf", binary=None)
    print(len(sections), len(tables))
