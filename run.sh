#!/usr/bin/env bash
# Retrolens launcher - uses the project venv so 'python' does not need to be on PATH.
cd "$(dirname "$0")" || exit 1

if [ ! -f "venv/Scripts/python.exe" ] && [ ! -f "venv/bin/python" ]; then
    echo "[ERROR] venv not found. Create it first with:"
    echo "    python3 -m venv venv   (macOS/Linux)"
    echo "    py -m venv venv        (Windows)"
    echo "Then install dependencies:"
    echo "    venv/bin/pip install -r requirements.txt       (macOS/Linux)"
    echo "    venv/Scripts/pip install -r requirements.txt   (Windows)"
    exit 1
fi

if [ -f "venv/Scripts/python.exe" ]; then
    PY="venv/Scripts/python.exe"
else
    PY="venv/bin/python"
fi

exec "$PY" main.py
