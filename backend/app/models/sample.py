"""Amostra de áudio."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlmodel import Field, SQLModel


class SampleSource(str, Enum):
    RECORDING = "recording"
    UPLOAD = "upload"
    SEED_TTS = "seed_tts"
    """Gerado por TTS. Bootstrap de pipeline — NUNCA entra em métrica de fala infantil."""


class Sample(SQLModel, table=True):
    __tablename__ = "sample"

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)

    label: str = ""
    """Nome curto para achar a amostra na UI."""
    source: SampleSource = SampleSource.UPLOAD
    original_filename: str | None = None

    audio_path: str | None = None
    """None depois do descarte por privacidade."""
    sha256: str = Field(index=True)
    duration_s: float = 0.0
    sample_rate: int = 16_000

    reference_text: str | None = None
    """Opcional. Quando presente, habilita alinhamento forçado, GOP e métricas de leitura."""

    sintetico: bool = Field(default=False, index=True)
    """True para áudio de TTS. Excluído das métricas de fala infantil."""
    tts_voice: str | None = None

    audio_discarded_at: datetime | None = None
    """Marcado quando DISCARD_AUDIO_AFTER_FEATURES apaga o WAV. Quebra o player
    sincronizado e impede re-execução — a UI avisa."""

    notes: str | None = None
