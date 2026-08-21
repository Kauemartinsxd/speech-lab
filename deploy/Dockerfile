# Imagem CPU-only. Não precisa de GPU: as duas camadas juntas levam poucos
# segundos por análise em 4 vCPU, e quem grava está esperando de qualquer jeito.
FROM python:3.11-slim

# espeak-ng  -> gera a pronúncia esperada (camada de sons)
# libsndfile -> lê o WAV que o navegador manda
# ffmpeg     -> decodifica qualquer outro formato enviado por upload
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng libespeak-ng1 libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# torch do índice CPU: o wheel padrão do PyPI arrasta ~2,5 GB de CUDA que aqui
# não serve para nada.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio \
 && pip install --no-cache-dir \
        transformers phonemizer fastapi "uvicorn[standard]" python-multipart \
        soundfile numpy huggingface_hub

ENV PYTHONUNBUFFERED=1 \
    ASR_DEVICE=cpu \
    HF_HOME=/models \
    PHONEMIZER_ESPEAK_LIBRARY=/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1

WORKDIR /app
COPY server/ ./server/
COPY static/ ./static/

EXPOSE 8765
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8765"]
