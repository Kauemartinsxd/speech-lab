"""Execução de um conjunto de engines sobre uma amostra."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class RunStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Run(SQLModel, table=True):
    __tablename__ = "run"

    id: int | None = Field(default=None, primary_key=True)
    sample_id: int = Field(foreign_key="sample.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    finished_at: datetime | None = None

    status: RunStatus = RunStatus.RUNNING
    engines_requested: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    options: dict = Field(default_factory=dict, sa_column=Column(JSON))
    """Overrides de runtime vindos da UI, ex.: threshold de GOP."""
