import re

from flask import Flask, request, jsonify
from urllib.parse import quote
import os, requests, sys
import traceback
import threading
import time
from queue import Queue
import fitz
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

app = Flask(__name__)

API_URL = os.getenv("PAPERLESS_API_URL", "http://webserver:8000")
TOKEN   = os.getenv("PAPERLESS_API_TOKEN")
paperless_headers = {"Authorization": f"Token {TOKEN}"}

LIBRETRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "http://libretranslate:5000")
LANG_SOURCE = os.getenv("LANG_SOURCE", 'de')
LANG_TARGET = os.getenv("LANG_TARGET", 'en')
Y_TOL = float(os.getenv("Y_TOL", 2.0))
FONTFILE = None # or pymupdf font file path
FIELD_ORIGINAL_DOC_NAME = 'Original document'
DOC_TYPE_TRANSLATION_NAME = 'Translation'

# In-memory queue for processing
translation_queue = Queue()
processing_stats = {"total": 0, "completed": 0, "failed": 0}

def clean_ocr_text(text):
    """Clean up common OCR errors before translation"""
    # Common OCR character substitutions
    ocr_fixes = {
        # Common character misreads
        'rn': 'm',  # rn often misread as m
        '|': 'l',   # pipe often misread as l

        # German-specific fixes
        'ü': 'ü', 'ä': 'ä', 'ö': 'ö', 'ß': 'ß',  # Ensure proper encoding
    }

    cleaned = text

    # Fix obvious OCR errors
    for wrong, right in ocr_fixes.items():
        cleaned = cleaned.replace(wrong, right)

    # Fix common quote marks
    cleaned = cleaned.replace('„', '"')  # German opening quote
    cleaned = cleaned.replace('"', '"')  # German closing quote

    # Smart paragraph handling for better translation
    # First clean up spaces and tabs
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)  # Remove leading spaces
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)  # Remove trailing spaces

    # Preserve paragraph breaks (double newlines) but merge lines within paragraphs
    paragraphs = re.split(r'\n\s*\n', cleaned)  # Split on paragraph breaks

    processed_paragraphs = []
    for paragraph in paragraphs:
        if paragraph.strip():
            # Within each paragraph, merge lines that don't end with sentence terminators
            lines = paragraph.strip().split('\n')
            merged_paragraph = []
            current_sentence = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if current_sentence:
                    # Check if previous line ended with sentence terminator
                    if current_sentence[-1] in '.!?:':
                        merged_paragraph.append(current_sentence)
                        current_sentence = line
                    else:
                        # Merge with previous line
                        current_sentence += " " + line
                else:
                    current_sentence = line

            # Add the last sentence
            if current_sentence:
                merged_paragraph.append(current_sentence)

            processed_paragraphs.append('\n'.join(merged_paragraph))

    cleaned = '\n\n'.join(processed_paragraphs)

    # Fix common word boundaries
    cleaned = re.sub(r'(\w)([A-Z])', r'\1 \2', cleaned)  # Add space before caps

    return cleaned.strip()


def translate(text, from_language):
    resp = requests.post(
        f"{LIBRETRANSLATE_URL}/translate",
        json={
            "q": text,
            "source": from_language or LANG_SOURCE,
            "target": LANG_TARGET,
            "format": "text"
        },
        timeout=60  # LibreTranslate is much faster
    )
    resp.raise_for_status()
    translation = resp.json().get("translatedText", "")
    return translation


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    block: int
    line: int
    word_no: int

@dataclass
class Line:
    bbox: Tuple[float, float, float, float]
    text: str

# ---------------------------
# Helpers
# ---------------------------
def debug_print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def estimate_fontsize_from_bbox(bbox: Tuple[float, float, float, float]) -> float:
    height = max(0.1, bbox[3] - bbox[1])
    # For redaction, use nearly full height for font size
    size = height * 0.95
    return max(6.0, min(128.0, size)) * 2

def union_bbox(b1, b2):
    return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))

def inflate_rect_tuple(bbox: Tuple[float, float, float, float], d: float) -> Tuple[float, float, float, float]:
    # Manual inflate (works across PyMuPDF versions)
    return (bbox[0] - d, bbox[1] - d, bbox[2] + d, bbox[3] + d)

# ---------------------------
# Extraction
# ---------------------------
def read_words(pdf: str, max_pages: int = None) -> List[Dict[str, Any]]:
    doc = fitz.open('pdf', pdf)
    pages = []
    total_pages = len(doc)
    if max_pages is not None:
        total_pages = min(total_pages, max_pages)

    debug_print_header(f"Step 1: Reading words (pages: {total_pages})")

    for i in range(total_pages):
        page = doc[i]
        words_raw = page.get_text("words")  # [x0, y0, x1, y1, "word", block, line, word_no]
        words = []
        for tup in words_raw:
            try:
                a, b, c, d, t, e, f, g = tup
                if t and str(t).strip():
                    words.append(Word(a, b, c, d, str(t), int(e), int(f), int(g)))
            except Exception as ex:
                print(f"[WARN] Malformed word tuple on page {i+1}: {tup} | {ex}")

        print(f"[INFO] Page {i+1}: size=({page.rect.width:.1f}x{page.rect.height:.1f}), words={len(words)}")
        if len(words) > 0:
            print(f"       Sample words: {', '.join(w.text for w in words[:8])} ...")
        pages.append({"page": i + 1, "width": page.rect.width, "height": page.rect.height, "words": words})
    return pages

def group_words_into_lines(words: List[Word], y_tol: float = 2.0) -> List[Line]:
    if not words:
        return []
    # Sort top-to-bottom, then left-to-right
    words_sorted = sorted(words, key=lambda w: (w.y0, w.x0))
    lines: List[List[Word]] = []
    current: List[Word] = []
    current_y = None

    for w in words_sorted:
        if current_y is None:
            current = [w]
            current_y = w.y0
            continue
        if abs(w.y0 - current_y) <= y_tol:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda k: k.x0))
            current = [w]
            current_y = w.y0
    if current:
        lines.append(sorted(current, key=lambda k: k.x0))

    output: List[Line] = []
    for idx, line_words in enumerate(lines, 1):
        text = " ".join(lw.text for lw in line_words).strip()
        if not text:
            continue
        bbox = (line_words[0].x0, line_words[0].y0, line_words[0].x1, line_words[0].y1)
        for lw in line_words[1:]:
            bbox = union_bbox(bbox, (lw.x0, lw.y0, lw.x1, lw.y1))
        output.append(Line(bbox=bbox, text=text))
    return output

def build_translation_plan(pages_words: List[Dict[str, Any]], y_tol: float, from_language: str) -> List[Dict[str, Any]]:
    debug_print_header("Step 2: Grouping words into lines & building translation plan")
    pages_lines = []
    total_lines = 0
    for pw in pages_words:
        lines = group_words_into_lines(pw["words"], y_tol=y_tol)
        print(f"[INFO] Page {pw['page']}: grouped {len(lines)} visual lines from {len(pw['words'])} words")

        lines_plan = []
        for idx, ln in enumerate(lines, 1):
            x0, y0, x1, y1 = ln.bbox
            lines_plan.append({
                "bbox": ln.bbox,
                "text": ln.text,
                "translation": translate(ln.text, from_language),
            })
        total_lines += len(lines_plan)
        pages_lines.append({"page": pw["page"], "lines": lines_plan})
    print(f"[INFO] Total lines across pages: {total_lines}")
    return pages_lines

def grow_rect(rect: fitz.Rect, factor: float = 1.1) -> fitz.Rect:
    """
    Expand a rect by the given factor (default 10%) without moving the origin (0,0).
    Only the top-right corner is scaled outward.
    """
    # Current width and height
    width = rect.width
    height = rect.height

    # New dimensions after scaling
    new_width = width * factor
    new_height = height * factor

    # Keep the rect's bottom-left (x0, y0) fixed
    return fitz.Rect(rect.x0, rect.y0, rect.x0 + new_width, rect.y0 + new_height)

# ---------------------------
# Painting / Writing
# ---------------------------
def redact_mode(in_pdf: str, plan: List[Dict[str, Any]], fontfile: str = None):
    debug_print_header("Step 2: Using redactions to replace text (REDACT mode)")
    doc: fitz.Document = fitz.open('pdf', in_pdf)
    for pinfo in plan:
        page: fitz.Page = doc[pinfo["page"] - 1]
        print(f"[PAGE {pinfo['page']}] adding redaction annots: {len(pinfo['lines'])}")
        for idx, ln in enumerate(pinfo["lines"], 1):
            rect = grow_rect(fitz.Rect(*ln["bbox"]), 2)
            fontsize = estimate_fontsize_from_bbox(ln["bbox"])
            print(f"    [REDACT] line#{idx}: rect=({rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f}) "
                  f"fontsize≈{fontsize:.1f} text={ln["translation"]}")
            page.add_redact_annot(
                rect,
                text=ln["translation"],
                fontname=None if fontfile else "helv",
                #fontfile=fontfile, # not supported in pip version, too old?
                fontsize=fontsize,
                fill=(1, 1, 1),
                align=0,
            )
        print(f"[PAGE {pinfo['page']}] applying redactions")
        try:
            page.apply_redactions()
        except Exception as ex:
            print(f"[ERROR] apply_redactions() failed on page {pinfo['page']}: {ex}")
            print("        Skipping overlay fallback (overlay mode removed)")

    return doc.tobytes(deflate=True, garbage=4)

def process_translation_job(doc_id, custom_field_id, doc_type_id):
    """Process a single translation job"""
    print(f"[translator] Processing translation for document {doc_id}", flush=True)
    try:
        # Fetch document content
        r = requests.get(
            f"{API_URL}/api/documents/{doc_id}/",
            headers=paperless_headers
        )
        r.raise_for_status()
        document = r.json()

        if document['document_type'] == doc_type_id:
            print(f"[translator] Document {doc_id} is already a translation, skipping translation", flush=True)
            processing_stats["completed"] += 1
            return

        original_text = document.get("content", "")

        if not original_text.strip():
            print(f"[translator] No content found for document {doc_id}", flush=True)
            return

        # Clean up common OCR errors before translation
        cleaned_text = clean_ocr_text(original_text)

        # Detect language - skip translation if already English
        detected_lang = None
        try:
            detected_lang = detect(cleaned_text)
            print(f"[translator] Detected language: {detected_lang} for document {doc_id}", flush=True)

            if detected_lang == 'en':
                print(f"[translator] Document {doc_id} is already in English, skipping translation", flush=True)
                processing_stats["completed"] += 1
                return

        except LangDetectException as e:
            print(f"[translator] Language detection failed for document {doc_id}: {e}, proceeding with translation", flush=True)

        # Step 0: get PDF bytes
        r = requests.get(
            f"{API_URL}/api/documents/{doc_id}/download/",
            headers=paperless_headers
        )
        r.raise_for_status()
        pdf_content = r.content
        # Step 1
        pages_words = read_words(pdf_content)
        # Step 2
        plan = build_translation_plan(pages_words, Y_TOL, detected_lang)
        # Step 3
        new_content = redact_mode(pdf_content, plan, FONTFILE)

        # Make new document
        upload = requests.post(f"{API_URL}/api/documents/post_document/", headers=paperless_headers, data={
            'title': f"{document['title']} ({LANG_TARGET})",
            'created': document['created'],
            'correspondent': document['correspondent'],
            'document_type': doc_type_id,
            'storage_path': document['storage_path'],
            'tags': document['tags'],
            'archive_serial_number': document['archive_serial_number'],
            'custom_fields': [],
        }, files={'document': new_content})
        upload.raise_for_status()

        # wait for translation document id
        task_id = upload.json()
        task = None
        while not task or task.get('status') != 'SUCCESS':
            task = requests.get(f"{API_URL}/api/tasks/", headers=paperless_headers, params={'task_id': task_id})
            task = task.json()[0]
            if task['status'] in ['FAILURE', 'REVOKED']:
                raise Exception(f"Task {task_id} failed: {task['result']}")
            if task['status'] != 'SUCCESS':
                print(f"waiting on {task_id}: {task['status']}", file=sys.stderr)
                time.sleep(0.5)

        # patch translation
        translation_doc_id = task['related_document']
        r = requests.patch(f"{API_URL}/api/documents/{translation_doc_id}/", headers=paperless_headers, json={
            'content': translate(document['content'], detected_lang),
            'custom_fields': [{'field': custom_field_id, 'value': [document['id']]}]
        })
        r.raise_for_status()

        processing_stats["completed"] += 1
        print(f"[translator] Successfully translated document {doc_id}", flush=True)

    except Exception as e:
        processing_stats["failed"] += 1
        print(f"[translator] Error translating document {doc_id}: {str(e)}", file=sys.stderr)
        if isinstance(e, requests.exceptions.HTTPError):
            print('HTTP Error:', e.response.content, e.response.headers, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

def queue_worker():
    """Background worker that processes translation jobs serially"""
    print("[translator] Queue worker started", flush=True)
    custom_field_id, doc_type_id = setup_ids()
    while True:
        try:
            # Get job from queue (blocks until available)
            doc_id = translation_queue.get(timeout=1)
            process_translation_job(doc_id, custom_field_id, doc_type_id)
            translation_queue.task_done()
        except:
            # Timeout or other error, continue
            time.sleep(0.1)

def setup_ids():
    r = requests.get(
        f"{API_URL}/api/custom_fields/?page_size=500&name__icontains={quote(FIELD_ORIGINAL_DOC_NAME)}",
        headers=paperless_headers
    )
    r.raise_for_status()
    existing_field = next((f['id'] for f in r.json()['results'] if f['data_type'] == 'documentlink'), None)
    if not existing_field:
        print('creating custom field')
        r = requests.post(
            f"{API_URL}/api/custom_fields/", headers=paperless_headers, json={
                "name": "Original document",
                "data_type": "documentlink",
            }
        )
        r.raise_for_status()
        existing_field = r.json()['id']

    r = requests.get(
        f"{API_URL}/api/document_types/?page_size=500&name__icontains={quote(DOC_TYPE_TRANSLATION_NAME)}",
        headers=paperless_headers
    )
    r.raise_for_status()
    existing_doc_type = next((f['id'] for f in r.json()['results'] if f['name'] == DOC_TYPE_TRANSLATION_NAME), None)
    if not existing_doc_type:
        print('creating document type')
        r = requests.post(
            f"{API_URL}/api/document_types/", headers=paperless_headers, json={
                "name": DOC_TYPE_TRANSLATION_NAME,
            }
        )
        r.raise_for_status()
        existing_doc_type = r.json()['id']

    print('custom field id:', existing_field, ', doc type id:', existing_doc_type,)
    return existing_field, existing_doc_type


@app.route("/", methods=["GET"])
def health():
    return "OK", 200

@app.route("/status", methods=["GET"])
def status():
    """Return queue status"""
    return jsonify({
        "queue_size": translation_queue.qsize(),
        "stats": processing_stats
    })

@app.route("/translate", methods=["POST"])
def queue_translation():
    """Queue a translation job (non-blocking)"""
    print("[translator] /translate endpoint hit", flush=True)
    try:
        data = request.json
        if not data or "document_id" not in data:
            return "Missing document_id", 400

        doc_id = data["document_id"]

        # Add to queue
        translation_queue.put(doc_id)
        processing_stats["total"] += 1

        print(f"[translator] Queued translation for document {doc_id}", flush=True)
        return jsonify({"status": "queued", "document_id": doc_id}), 202

    except Exception as e:
        print("[translator] Error in queue_translation:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return str(e), 500

if __name__ == "__main__":
    worker_thread = threading.Thread(target=queue_worker, daemon=True)
    worker_thread.start()
    app.run(host="0.0.0.0", port=5000)