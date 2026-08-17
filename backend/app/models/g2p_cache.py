"""Cache de G2P por palavra.

O phonemizer sobe um processo do espeak a cada chamada; cachear por palavra faz
o estágio 1 do phoneme_gop deixar de ser o gargalo.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class G2pCacheEntry(SQLModel, table=True):
    __tablename__ = "g2p_cache"

    id: int | None = Field(default=None, primary_key=True)
    word: str = Field(index=True)
    language: str = "pt-br"
    backend: str = "espeak"
    backend_version: str | None = None
    """Invalida o cache quando o espeak muda de versão."""

    ipa: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
