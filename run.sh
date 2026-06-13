#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PY=python3
VENV=.venv

if [ ! -d "$VENV" ]; then
  echo "==> Creating virtualenv ($VENV)"
  $PY -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "==> Installing dependencies (first run downloads models, ~2-3 min)"
pip install -q --upgrade pip
pip install -q -r requirements.txt

if ! compgen -G "data/pdfs/*.pdf" > /dev/null; then
  echo "==> Generating sample PDF corpus (10 docs x 210 pages)"
  python gen_sample_pdfs.py
fi

echo ""
echo "==> Starting server at http://localhost:8000"
echo "    (In the UI, click 'Ingest data/pdfs/' to load the corpus.)"
echo ""
exec uvicorn app:app --host 0.0.0.0 --port 8000
