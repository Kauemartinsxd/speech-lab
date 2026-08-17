"""Modelos SQLModel.

Todos precisam ser importados aqui antes do create_all — senão o SQLModel não
registra a tabela e o banco nasce incompleto.
"""

from app.models.annotation import Annotation
from app.models.g2p_cache import G2pCacheEntry
from app.models.result import EngineOutcome
from app.models.run import Run, RunStatus
from app.models.sample import Sample, SampleSource

__all__ = [
    "Annotation",
    "EngineOutcome",
    "G2pCacheEntry",
    "Run",
    "RunStatus",
    "Sample",
    "SampleSource",
]
