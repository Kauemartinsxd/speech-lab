from __future__ import annotations

import shutil

from fastapi import APIRouter

from app.config import get_settings
from app.engines.registry import engine_matrix
from app.runtime import device
from app.runtime.model_cache import get_model_cache

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "device": device.describe(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "espeak_ng": shutil.which("espeak-ng") is not None,
        "modelos_residentes": get_model_cache().resident(),
        "privacidade": {
            "descarta_audio_apos_features": settings.discard_audio_after_features,
        },
    }


@router.get("/engines")
def engines() -> dict:
    """Matriz de engines para a UI. Barato de propósito: não carrega modelo."""
    return {"engines": engine_matrix()}


@router.post("/engines/unload")
def unload() -> dict:
    """Libera todos os modelos da memória. Útil em 16 GB entre experimentos."""
    get_model_cache().unload_all()
    return {"status": "ok", "modelos_residentes": get_model_cache().resident()}
