#!/usr/bin/env bash
set -euo pipefail

cd /Users/gunn/Documents/nm_i_ai/tripletex

VENV_PYTHON=".venv/bin/python"
VENV_UVICORN=".venv/bin/uvicorn"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Fant ikke $VENV_PYTHON"
  echo "Lag virtualenv først."
  exit 1
fi

if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:8000}"
LOCAL_DOCS_URL="${LOCAL_BASE_URL}/docs"
SUBMISSION_URL="https://app.ainm.no/submit/tripletex"
TRIPLETEX_UI_URL=""

if [ -n "${TRIPLETEX_BASE_URL:-}" ]; then
  TRIPLETEX_UI_URL="${TRIPLETEX_BASE_URL%/v2}"
fi

open_url() {
  local url="$1"
  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  else
    echo "Åpne manuelt: $url"
  fi
}

echo "Python:"
"$VENV_PYTHON" --version

echo
echo "Kjører tester..."
"$VENV_PYTHON" -m pytest -q

echo
echo "Åpner relevante URL-er..."
open_url "$SUBMISSION_URL"

if [ -n "$TRIPLETEX_UI_URL" ]; then
  open_url "$TRIPLETEX_UI_URL"
fi

(
  sleep 3
  open_url "$LOCAL_BASE_URL"
  open_url "$LOCAL_DOCS_URL"
) &

echo
echo "Starter API på ${LOCAL_BASE_URL}"
exec "$VENV_UVICORN" main:app --host 127.0.0.1 --port 8000 --reload
