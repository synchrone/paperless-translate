from flask import Flask, request, jsonify
import os, requests, sys
import traceback
import threading
import time
import redis
import json
import re
from queue import Queue
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

app = Flask(__name__)

API_URL = "http://webserver:8000"
TOKEN   = "[insert token here]"
LIBRETRANSLATE_URL = "http://libretranslate:5000"

# Redis connection for job queue
redis_client = redis.Redis(host='broker', port=6379, db=0, decode_responses=True)

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

def process_translation_job(doc_id):
    """Process a single translation job"""
    print(f"[translator] Processing translation for document {doc_id}", flush=True)
    try:
        # Fetch OCR text
        r = requests.get(
            f"{API_URL}/api/documents/{doc_id}/",
            headers={"Authorization": f"Token {TOKEN}"}
        )
        r.raise_for_status()
        text = r.json().get("content", "")
        
        if not text.strip():
            print(f"[translator] No content found for document {doc_id}", flush=True)
            return
            
        # Clean up common OCR errors before translation
        cleaned_text = clean_ocr_text(text)
        
        # Detect language - skip translation if already English
        try:
            detected_lang = detect(cleaned_text)
            print(f"[translator] Detected language: {detected_lang} for document {doc_id}", flush=True)
            
            if detected_lang == 'en':
                print(f"[translator] Document {doc_id} is already in English, skipping translation", flush=True)
                processing_stats["completed"] += 1
                return
                
        except LangDetectException as e:
            print(f"[translator] Language detection failed for document {doc_id}: {e}, proceeding with translation", flush=True)
        
        # Request translation from LibreTranslate
        resp = requests.post(
            f"{LIBRETRANSLATE_URL}/translate",
            json={
                "q": cleaned_text,
                "source": "de",
                "target": "en",
                "format": "text"
            },
            timeout=60  # LibreTranslate is much faster
        )
        resp.raise_for_status()
        translation = resp.json().get("translatedText", "")
        
        # Patch back to Paperless
        new_content = f"{text}\n\n-----------------\n\n{translation}"
        patch = requests.patch(
            f"{API_URL}/api/documents/{doc_id}/",
            headers={"Authorization": f"Token {TOKEN}"},
            json={"content": new_content}
        )
        patch.raise_for_status()
        
        processing_stats["completed"] += 1
        print(f"[translator] Successfully translated document {doc_id}", flush=True)
        
    except Exception as e:
        processing_stats["failed"] += 1
        print(f"[translator] Error translating document {doc_id}: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

def queue_worker():
    """Background worker that processes translation jobs serially"""
    print("[translator] Queue worker started", flush=True)
    while True:
        try:
            # Get job from queue (blocks until available)
            doc_id = translation_queue.get(timeout=1)
            process_translation_job(doc_id)
            translation_queue.task_done()
        except:
            # Timeout or other error, continue
            time.sleep(0.1)

# Start background worker thread
worker_thread = threading.Thread(target=queue_worker, daemon=True)
worker_thread.start()

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
    app.run(host="0.0.0.0", port=5000)