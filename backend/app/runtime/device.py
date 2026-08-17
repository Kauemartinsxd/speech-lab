"""Seleção de device.

Regra do lab (ver README): o forward do wav2vec2 pode ir para MPS, mas o
alinhamento forçado e o cálculo de GOP ficam em CPU. O `forced_align` do
torchaudio não tem kernel MPS, e o GOP produz números que serão reportados —
precisa ser determinístico.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings


@lru_cache
def pick_device() -> str:
    """Resolve o device de inferência: auto => cuda -> mps -> cpu."""
    configured = get_settings().device.lower()
    if configured != "auto":
        return configured

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def alignment_device() -> str:
    """Device do estágio de alinhamento/GOP. CPU salvo override explícito."""
    if get_settings().force_cpu_for_alignment:
        return "cpu"
    return pick_device()


def describe() -> dict[str, object]:
    """Diagnóstico para /health e para o make check."""
    info: dict[str, object] = {
        "inference_device": pick_device(),
        "alignment_device": alignment_device(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        info["torch_version"] = None
    return info
