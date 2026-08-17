"""Saída de uma engine dentro de uma execução.

Guarda o EngineResult inteiro em `payload`: sem isso não dá para conferir um
número meses depois sem re-executar o modelo.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class EngineOutcome(SQLModel, table=True):
    __tablename__ = "engine_outcome"

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True)
    sample_id: int = Field(foreign_key="sample.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    engine: str = Field(index=True)
    status: str
    unavailable_reason: str | None = None

    transcript: str | None = None
    latency_ms: int = 0
    cost_estimate_usd: float | None = None
    model_version: str | None = None
    device: str | None = None

    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    """EngineResult completo serializado, para auditoria."""
