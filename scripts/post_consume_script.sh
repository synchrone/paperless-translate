#!/bin/bash

# Get document ID from environment
DOC_ID="${DOCUMENT_ID}"

if [ -z "$DOC_ID" ]; then
    echo "Error: DOCUMENT_ID not found in environment"
    exit 1
fi

echo "****************************************************************************************************"
echo "Post-consume script starting for document: $DOC_ID"
echo "****************************************************************************************************"

# Translator service URL (will be in same Docker network)
TRANSLATOR_URL="http://translator:5000"

# Send translation request to translator service (non-blocking)
response=$(curl -s -w "%{http_code}" -X POST \
    "$TRANSLATOR_URL/translate" \
    -H "Content-Type: application/json" \
    -d "{\"document_id\": \"$DOC_ID\"}" \
    --max-time 5 \
    -o /tmp/translate_response.txt)

http_code="${response: -3}"

if [ "$http_code" = "202" ]; then
    echo "Document $DOC_ID queued for translation"
    echo "****************************************************************************************************"
    echo "Translation queued: Document $DOC_ID added to translation queue"
    echo "****************************************************************************************************"
else
    echo "Failed to queue translation: HTTP $http_code"
    cat /tmp/translate_response.txt 2>/dev/null || true
    echo "Document processing will continue without translation"
fi

echo "Post-consume script completed"