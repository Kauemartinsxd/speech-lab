"""Contrato das engines.

A promessa do lab é que uma engine caída não derruba as outras. Isso só é
verdade se `run()` nunca levantar e `check_availability()` nunca levantar nem
carregar modelo. Estes testes são o que impede a promessa de virar folclore.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.engines.base import (
    AudioInput,
    Engine,
    EngineRequest,
    EngineResult,
    EngineStatus,
)
from app.engines.registry import ENGINE_SPECS, engine_matrix, load_engine

IMPLEMENTADAS = [name for name in ENGINE_SPECS if load_engine(name) is not None]


def _req(path: Path) -> EngineRequest:
    return EngineRequest(
        audio=AudioInput(path=path, sample_rate=16_000, duration_s=1.0, sha256="0" * 64),
        reference_text=None,
        options={},
    )


@pytest.mark.parametrize("name", IMPLEMENTADAS)
def test_check_availability_nao_levanta(name: str):
    engine = load_engine(name)
    av = engine.check_availability()
    assert isinstance(av.available, bool)
    if not av.available:
        assert av.reason, "engine indisponível tem que dizer o motivo"


@pytest.mark.parametrize("name", IMPLEMENTADAS)
def test_check_availability_nao_carrega_modelo(name: str):
    """Barato de propósito: a matriz da UI é montada antes de qualquer execução."""
    from app.runtime.model_cache import get_model_cache

    cache = get_model_cache()
    antes = set(cache.resident())
    load_engine(name).check_availability()
    assert set(cache.resident()) == antes


@pytest.mark.parametrize("name", IMPLEMENTADAS)
async def test_run_nao_levanta_com_audio_invalido(name: str, tmp_path: Path):
    """O caminho infeliz: arquivo que não existe. A engine tem que devolver
    ERROR ou UNAVAILABLE, nunca propagar exceção para o runner."""
    engine = load_engine(name)
    fantasma = tmp_path / "nao-existe.wav"

    result = await engine.run(_req(fantasma))

    assert isinstance(result, EngineResult)
    assert result.status in (EngineStatus.ERROR, EngineStatus.UNAVAILABLE)
    assert result.unavailable_reason
    assert result.engine == name


@pytest.mark.parametrize("name", IMPLEMENTADAS)
def test_capabilities_declaradas(name: str):
    caps = load_engine(name).capabilities
    for campo in (
        "produces_orthographic",
        "sends_audio_externally",
        "is_reproducible",
        "normalizes_to_standard",
    ):
        assert isinstance(getattr(caps, campo), bool)


def test_whisper_declara_que_normaliza():
    """O selo de controle negativo na UI depende deste flag."""
    assert load_engine("whisper_baseline").capabilities.normalizes_to_standard is True


def test_ctc_declara_que_nao_normaliza():
    assert load_engine("ctc_greedy").capabilities.normalizes_to_standard is False


def test_engines_locais_nao_mandam_audio_pra_fora():
    for name in ("whisper_baseline", "whisper_strict", "ctc_greedy"):
        assert load_engine(name).capabilities.sends_audio_externally is False


def test_matriz_sobrevive_a_engine_nao_implementada():
    """Durante a construção do lab há engines que ainda não existem. A matriz
    tem que listá-las como não implementadas em vez de quebrar."""
    matriz = {e["name"]: e for e in engine_matrix()}
    assert set(matriz) == set(ENGINE_SPECS)
    for entrada in matriz.values():
        if not entrada["implemented"]:
            assert entrada["available"] is False
            assert entrada["reason"]


def test_toda_engine_implementada_e_subclasse():
    for name in IMPLEMENTADAS:
        assert isinstance(load_engine(name), Engine)
