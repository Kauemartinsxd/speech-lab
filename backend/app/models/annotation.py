"""Anotação humana — o ground truth.

Sem isto o lab não decide nada: WER, PER e F1 de detecção são todos medidos
contra esta tabela, nunca contra a saída de outra engine.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class Annotation(SQLModel, table=True):
    __tablename__ = "annotation"

    id: int | None = Field(default=None, primary_key=True)
    sample_id: int = Field(foreign_key="sample.id", index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    transcript_fiel: str
    """O que a pessoa realmente falou, em ortografia. "pobrema", não "problema"."""
    ipa: str | None = None
    """Transcrição fonética, opcional."""

    deviations: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    """Desvios marcados à mão: {expected, observed, type, category, process}."""

    annotator: str = "humano"
    notes: str | None = None
