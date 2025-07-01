This configuration will automatically translate your documents using LibreTranslate to provide automatic, entirely local translation of your documents. 

Setup:
1. Generate an API key in paperless-ngx under My Profile.
2. Edit `translator-queue.yml` to update with the appropriate token
3. Also in `translator-queue.yml`, update the `"source": "de",` line to the appropriate source language.

This will:
- When a new document is consumed, run the post_consume_script.sh which
- Triggers the translator-queue.py which cleans up some common OCR mistakes and queues a translation job
- When complete, appends the translated text to the "Content" field of the document.
- If the language is detected as English, translation will be skipped, and a note appended instead.
