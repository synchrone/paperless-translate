This configuration will automatically translate your documents using LibreTranslate to provide automatic, entirely local translation of your documents. 

Setup:
1. Generate an API key in paperless-ngx under My Profile.
2. Create a `.env` file in the root of the project.
3. Add the following line to the `.env` file, replacing `your_token_here` with the API key you generated:
   ```
   PAPERLESS_API_TOKEN=your_token_here
   ```
4. Also in `translator-queue.yml`, update the `"source": "de",` line to the appropriate source language.

This will:
- When a new document is consumed, run the post_consume_script.sh which
- Triggers the translator-queue.py which cleans up some common OCR mistakes and queues a translation job
- When complete, appends the translated text to the "Content" field of the document.
- If the language is detected as English, translation will be skipped, and a note appended instead.

This approach is heavily inspired by an [implementation](https://github.com/paperless-ngx/paperless-ngx/discussions/269#discussioncomment-12303929) by [kavishdahekar](https://github.com/kavishdahekar).

## Manual Translation & Queue Management

You can manually trigger a translation for a document or check the status of the translation queue using the following methods:

### Trigger a Translation

To manually trigger a translation for a specific document, send a POST request to the `/translate` endpoint with the document's ID. Replace `YOUR_DOCUMENT_ID` with the actual ID of the document you want to translate.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"document_id": YOUR_DOCUMENT_ID}' http://localhost:5000/translate
```

### Check Queue Status

To check the current status of the translation queue, including total, completed, and failed jobs, send a GET request to the `/status` endpoint:

```bash
curl http://localhost:5000/status
```
