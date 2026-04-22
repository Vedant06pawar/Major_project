"""
document_parser.py
==================
Parses PDF and DOCX files into a sequential list of blocks:

    [
        {"type": "text",  "content": "..."},
        {"type": "image", "image_path": "/tmp/adr_xxx/img_2.png"},
        ...
    ]

Blocks preserve document reading order so the TTS engine can narrate them
sequentially — text is read aloud, images are captioned and read aloud.
"""

import os
import io
import logging
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# Minimum image dimensions — smaller blobs are likely decorative bullets/icons
MIN_IMG_WIDTH  = 80
MIN_IMG_HEIGHT = 80


# ────────────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────────────

def parse_document(doc_path: str) -> list[dict]:
    """
    Dispatch to the correct parser based on file extension.

    Returns a list of block dicts in reading order.
    """
    ext = Path(doc_path).suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(doc_path)
    elif ext in {".docx", ".doc"}:
        return _parse_docx(doc_path)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}")


# ────────────────────────────────────────────────────────────────────────────
# PDF parser
# ────────────────────────────────────────────────────────────────────────────

def _parse_pdf(pdf_path: str) -> list[dict]:
    """
    Uses PyMuPDF (fitz) to iterate pages.
    Each page is split into text blocks and image blocks in the order
    they appear on the page (sorted by vertical position).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with: pip install pymupdf"
        ) from exc

    tmpdir = os.path.dirname(pdf_path)
    blocks: list[dict] = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        page_blocks: list[tuple[float, dict]] = []  # (y0, block_dict)

        # ── Text blocks ──────────────────────────────────────────────────────
        text_dict = page.get_text("blocks")  # list of (x0,y0,x1,y1,text,bno,btype)
        for tb in text_dict:
            x0, y0, x1, y1, text, bno, btype = tb
            if btype == 0 and text.strip():  # 0 = text block
                page_blocks.append((y0, {"type": "text", "content": text.strip()}))

        # ── Image blocks ─────────────────────────────────────────────────────
        img_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception as exc:
                log.warning("Page %d, img %d: extract failed: %s", page_num, img_idx, exc)
                continue

            img_bytes  = base_image["image"]
            img_ext    = base_image.get("ext", "png")
            img_smask  = base_image.get("smask", 0)

            # Skip masks (they are not standalone images)
            if img_smask and img_smask == xref:
                continue

            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as exc:
                log.warning("Page %d, img %d: PIL open failed: %s", page_num, img_idx, exc)
                continue

            w, h = pil_img.size
            if w < MIN_IMG_WIDTH or h < MIN_IMG_HEIGHT:
                log.debug("Page %d, img %d: skipped (too small %dx%d)", page_num, img_idx, w, h)
                continue

            # Find bounding box on the page to get y-position for ordering
            bbox = page.get_image_bbox(img_info)
            y0 = float(bbox.y0) if bbox else float("inf")

            img_filename = f"page{page_num}_img{img_idx}.{img_ext}"
            img_path     = os.path.join(tmpdir, img_filename)
            pil_img.save(img_path)

            page_blocks.append((y0, {"type": "image", "image_path": img_path}))

        # Sort by vertical position to preserve reading order
        page_blocks.sort(key=lambda t: t[0])
        blocks.extend(b for _, b in page_blocks)

    page_count = len(doc)
    doc.close()
    log.info("PDF parsed: %d pages, %d blocks total.", page_count, len(blocks))
    return blocks


# ────────────────────────────────────────────────────────────────────────────
# DOCX parser
# ────────────────────────────────────────────────────────────────────────────

def _parse_docx(docx_path: str) -> list[dict]:
    """
    Uses python-docx to walk through the document body XML in order.
    Paragraphs → text blocks.
    Inline images (r:drawing / r:pict) → image blocks.

    We iterate the raw XML children of the body element to preserve
    the exact ordering of paragraphs and drawing elements.
    """
    try:
        import docx
        from docx.oxml.ns import qn
        from lxml import etree
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX parsing. "
            "Install it with: pip install python-docx"
        ) from exc

    tmpdir   = os.path.dirname(docx_path)
    blocks: list[dict] = []
    img_counter = 0

    doc_obj  = docx.Document(docx_path)
    body     = doc_obj.element.body

    # Map relationship IDs to image bytes in the package
    part = doc_obj.part

    def _save_image_from_rId(rId: str) -> str | None:
        """Extract image bytes from a relationship ID and save to tmpdir."""
        nonlocal img_counter
        try:
            img_part = part.related_parts[rId]
            img_bytes = img_part.blob
            img_ext   = img_part.content_type.split("/")[-1]
            if img_ext in {"jpeg", "jpg"}:
                img_ext = "jpg"
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil_img.size
            if w < MIN_IMG_WIDTH or h < MIN_IMG_HEIGHT:
                return None
            fname = os.path.join(tmpdir, f"docx_img{img_counter}.{img_ext}")
            pil_img.save(fname)
            img_counter += 1
            return fname
        except Exception as exc:
            log.warning("DOCX image extraction failed (rId=%s): %s", rId, exc)
            return None

    def _extract_images_from_element(elem) -> list[str]:
        """Recursively find all blip rIds inside a drawing or pict element."""
        paths = []
        # a:blip r:embed="rId..."
        for blip in elem.findall(".//" + qn("a:blip")):
            rId = blip.get(qn("r:embed"))
            if rId:
                p = _save_image_from_rId(rId)
                if p:
                    paths.append(p)
        # v:imagedata r:id="rId..."  (legacy VML)
        for imgdata in elem.findall(".//" + qn("v:imagedata")):
            rId = imgdata.get(qn("r:id"))
            if rId:
                p = _save_image_from_rId(rId)
                if p:
                    paths.append(p)
        return paths

    # Walk body children in document order
    for child in body:
        tag = child.tag

        if tag == qn("w:p"):
            # ── Paragraph ───────────────────────────────────────────────────
            # Collect text from all runs first
            text_parts: list[str] = []
            for run in child.findall(".//" + qn("w:t")):
                t = run.text or ""
                text_parts.append(t)
            text = "".join(text_parts).strip()

            # Collect inline images embedded in this paragraph
            img_paths: list[str] = []
            for drawing in child.findall(".//" + qn("w:drawing")):
                img_paths.extend(_extract_images_from_element(drawing))
            for pict in child.findall(".//" + qn("w:pict")):
                img_paths.extend(_extract_images_from_element(pict))

            if text:
                blocks.append({"type": "text", "content": text})
            for p in img_paths:
                blocks.append({"type": "image", "image_path": p})

        elif tag == qn("w:tbl"):
            # ── Table → flatten to text ──────────────────────────────────────
            rows: list[str] = []
            for row in child.findall(".//" + qn("w:tr")):
                cells: list[str] = []
                for cell in row.findall(".//" + qn("w:tc")):
                    cell_texts = [
                        (t.text or "")
                        for t in cell.findall(".//" + qn("w:t"))
                    ]
                    cells.append(" ".join(cell_texts).strip())
                rows.append(", ".join(cells))
            table_text = "Table: " + "; ".join(rows)
            if table_text.strip():
                blocks.append({"type": "text", "content": table_text})

    log.info("DOCX parsed: %d blocks total.", len(blocks))
    return blocks
