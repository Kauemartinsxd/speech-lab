# Artefato de reprodutibilidade. O desenvolvimento do dia a dia é nativo com
# uv: Docker no macOS é uma VM Linux sem passagem de Metal, então dentro do
# container não existe MPS e tudo cai para CPU. Este Dockerfile existe para
# rodar o lab numa máquina Linux com NVIDIA, ou para congelar o ambiente.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        espeak-ng \
        libespeak-ng1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# no Debian a lib fica noutro caminho que no Homebrew
ENV PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY backend/ ./backend/
COPY scripts/ ./scripts/
COPY data/variantes_ptbr.yaml data/seed/ ./data/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/backend \
    HF_HOME=/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
