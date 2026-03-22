
#!/usr/bin/env bash
set -euo pipefail

cd /Users/gunn/Documents/nm_i_ai/tripletex

VENV_PYTHON=".venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Fant ikke $VENV_PYTHON"
  echo "Lag virtualenv først."
  exit 1
fi

echo "Python:"
"$VENV_PYTHON" --version

echo
echo "Kjører tester..."
"$VENV_PYTHON" -m pytest -q
