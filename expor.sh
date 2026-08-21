#!/usr/bin/env bash
# Expõe a ferramenta na internet via Cloudflare Tunnel, COM senha.
# Uso: ./expor.sh [porta]          (a senha é gerada e mostrada no fim)
#      TTS_PASSWORD=minhasenha ./expor.sh
set -euo pipefail
cd "$(dirname "$0")"
PORT="${1:-8765}"
VENV="${TTS_VENV:-$HOME/.venvs/tts}"
LOG_DIR="${TMPDIR:-/tmp}/tts-expor"; mkdir -p "$LOG_DIR"

command -v cloudflared >/dev/null || { echo "!! falta cloudflared: brew install cloudflared"; exit 1; }

# senha: usa a informada ou gera uma legível
if [ -z "${TTS_PASSWORD:-}" ]; then
  TTS_PASSWORD="$("$VENV/bin/python" -c "
import secrets
a='sol lua rio mar céu paz flor luz vento chuva pedra folha'.split()
print('-'.join(secrets.choice(a) for _ in range(3)) + str(secrets.randbelow(90)+10))")"
fi
export TTS_PASSWORD
export TTS_USER="${TTS_USER:-prof}"

# servidor (só em 127.0.0.1 — quem fala com o mundo é o túnel)
if ! curl -s -m 2 -o /dev/null "http://127.0.0.1:$PORT/api/status"; then
  echo ">> subindo o servidor..."
  nohup ./run.sh "$PORT" > "$LOG_DIR/server.log" 2>&1 &
else
  echo "!! já existe algo na porta $PORT — reinicie para a senha valer:"
  echo "   pkill -f 'uvicorn server.app' && ./expor.sh $PORT"
  exit 1
fi

echo ">> carregando os modelos (pode levar ~40 s na primeira vez)..."
until curl -s -m 2 "http://127.0.0.1:$PORT/api/status" -u "$TTS_USER:$TTS_PASSWORD" | grep -q '"fon":{"loaded":true'; do sleep 3; done

echo ">> abrindo o túnel..."
nohup cloudflared tunnel --url "http://localhost:$PORT" > "$LOG_DIR/tunnel.log" 2>&1 &
URL=""
for _ in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_DIR/tunnel.log" | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
[ -z "$URL" ] && { echo "!! túnel não subiu; veja $LOG_DIR/tunnel.log"; exit 1; }

cat <<FIM

  ┌─────────────────────────────────────────────────────
  │  $URL
  │  usuário: $TTS_USER     senha: $TTS_PASSWORD
  └─────────────────────────────────────────────────────

  A URL muda toda vez que o túnel reinicia.
  Para derrubar tudo:  pkill -f cloudflared; pkill -f 'uvicorn server.app'
  Logs: $LOG_DIR
FIM
