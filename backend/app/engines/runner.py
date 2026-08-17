"""Execução das engines com resultado streamando conforme cada uma termina.

Duas regras de concorrência:

- engines locais (torch, whisper) disputam um semáforo: em 16 GB, rodar cinco
  modelos de verdade ao mesmo tempo estoura a máquina;
- engines remotas (azure, gemini) não entram no semáforo — são I/O, não RAM.

Nenhuma engine derruba as outras: o contrato diz que `run` não levanta, e aqui
ainda há um cinto de segurança para o caso de uma implementação quebrar o
contrato.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from app.config import get_settings
from app.engines.base import (
    Engine,
    EngineRequest,
    EngineResult,
    EngineStatus,
    unavailable_result,
)

log = logging.getLogger(__name__)


async def _run_one(
    engine: Engine,
    req: EngineRequest,
    semaphore: asyncio.Semaphore | None,
    timeout_s: int,
) -> EngineResult:
    started = time.perf_counter()

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    try:
        av = engine.check_availability()
        if not av.available:
            result = unavailable_result(engine.name, av)
            result.latency_ms = _elapsed_ms()
            return result

        if semaphore is not None:
            async with semaphore:
                result = await asyncio.wait_for(engine.run(req), timeout=timeout_s)
        else:
            result = await asyncio.wait_for(engine.run(req), timeout=timeout_s)

        if not result.latency_ms:
            result.latency_ms = _elapsed_ms()
        return result

    except TimeoutError:
        log.warning("engine %s estourou o timeout de %ss", engine.name, timeout_s)
        return EngineResult(
            engine=engine.name,
            status=EngineStatus.TIMEOUT,
            unavailable_reason=f"excedeu {timeout_s}s",
            latency_ms=_elapsed_ms(),
        )
    except Exception as exc:
        # O contrato diz que run() não levanta. Se chegou aqui, a implementação
        # quebrou o contrato — registra e segue, sem contaminar as outras.
        log.exception("engine %s quebrou o contrato e levantou", engine.name)
        return EngineResult(
            engine=engine.name,
            status=EngineStatus.ERROR,
            unavailable_reason=f"{type(exc).__name__}: {exc}",
            latency_ms=_elapsed_ms(),
        )


async def run_engines(
    engines: list[Engine],
    req: EngineRequest,
) -> AsyncIterator[EngineResult]:
    """Dispara todas as engines e entrega cada resultado assim que fica pronto.

    Não espera a mais lenta: o consumidor (SSE) repassa para a UI conforme chega.
    """
    settings = get_settings()
    local_sem = asyncio.Semaphore(settings.max_local_concurrency)

    tasks = [
        asyncio.create_task(
            _run_one(
                engine,
                req,
                None if engine.capabilities.sends_audio_externally else local_sem,
                settings.engine_timeout_s,
            ),
            name=f"engine:{engine.name}",
        )
        for engine in engines
    ]

    try:
        for coro in asyncio.as_completed(tasks):
            yield await coro
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
