"""Cache LRU de modelos carregados.

Em 16 GB de RAM isto não é higiene, é requisito: whisper large-v3 int8 ocupa
~1.5 GB e cada wav2vec2-large ~1.2 GB. Carregar tudo no import estoura a
máquina — daí a carga sob demanda com liberação explícita.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)


class ModelCache:
    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, max_size)
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.RLock()

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        """Devolve o modelo em cache ou carrega, despejando o mais antigo se preciso.

        O carregamento acontece com o lock segurado de propósito: duas engines
        pedindo o mesmo modelo simultaneamente carregariam duas cópias e
        estourariam a memória.
        """
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]

            while len(self._items) >= self._max_size:
                evicted_key, evicted = self._items.popitem(last=False)
                log.info("despejando modelo do cache: %s", evicted_key)
                self._release(evicted)

            log.info("carregando modelo: %s", key)
            model = loader()
            self._items[key] = model
            return model

    def unload(self, key: str) -> bool:
        with self._lock:
            model = self._items.pop(key, None)
            if model is None:
                return False
            log.info("liberando modelo: %s", key)
            self._release(model)
            return True

    def unload_all(self) -> None:
        with self._lock:
            for key, model in list(self._items.items()):
                log.info("liberando modelo: %s", key)
                self._release(model)
            self._items.clear()

    def resident(self) -> list[str]:
        with self._lock:
            return list(self._items.keys())

    @staticmethod
    def _release(model: Any) -> None:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass


_cache: ModelCache | None = None
_cache_lock = threading.Lock()


def get_model_cache() -> ModelCache:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = ModelCache(get_settings().model_cache_size)
        return _cache
