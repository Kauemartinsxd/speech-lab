#!/usr/bin/env bash
# Sobe a ferramenta de análise de fala. Uso: ./run.sh [porta]
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8765}"
VENV="${TTS_VENV:-$HOME/.venvs/tts}"   # fora do iCloud de propósito (venv com 20k arquivos)

if [ ! -x "$VENV/bin/python" ]; then
  echo ">> criando ambiente em $VENV"
  command -v uv >/dev/null || brew install uv
  uv venv "$VENV" --python 3.11
  uv pip install --python "$VENV/bin/python" -r requirements.txt
fi
command -v espeak-ng >/dev/null || { echo ">> instalando espeak-ng"; brew install espeak-ng; }

echo ">> http://localhost:$PORT   (Ctrl+C para parar)"
exec "$VENV/bin/python" -m uvicorn server.app:app --host 127.0.0.1 --port "$PORT"
