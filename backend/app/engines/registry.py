"""Descoberta e habilitação de engines.

Os módulos concretos são importados sob demanda: importar `ctc_greedy` puxa
transformers, e a matriz de engines precisa ser montada sem pagar esse custo.
"""

from __future__ import annotations

import importlib
import logging

from app.config import get_settings
from app.engines.base import Availability, Engine

log = logging.getLogger(__name__)

# nome -> (módulo, classe). A ordem aqui é a ordem das colunas na UI:
# o controle negativo primeiro, para o contraste ficar imediato.
ENGINE_SPECS: dict[str, tuple[str, str]] = {
    "whisper_baseline": ("app.engines.whisper_baseline", "WhisperBaselineEngine"),
    "whisper_strict": ("app.engines.whisper_baseline", "WhisperStrictEngine"),
    "ctc_greedy": ("app.engines.ctc_greedy", "CtcGreedyEngine"),
    "phoneme_gop": ("app.engines.phoneme_gop.engine", "PhonemeGopEngine"),
    "azure_pa": ("app.engines.azure_pa", "AzurePronunciationEngine"),
    "gemini_audio": ("app.engines.gemini_audio", "GeminiAudioEngine"),
}

_instances: dict[str, Engine] = {}


def load_engine(name: str) -> Engine | None:
    """Instancia a engine. Devolve None se o módulo nem existir ainda.

    Um módulo ausente é estado normal durante a construção do lab (as engines
    entram por milestone) e não pode derrubar a matriz.
    """
    if name in _instances:
        return _instances[name]

    spec = ENGINE_SPECS.get(name)
    if spec is None:
        return None

    module_path, class_name = spec
    try:
        module = importlib.import_module(module_path)
        engine = getattr(module, class_name)()
    except (ImportError, AttributeError) as exc:
        log.info("engine %s não carregada: %s", name, exc)
        return None

    _instances[name] = engine
    return engine


def enabled_engine_names() -> list[str]:
    flags = get_settings().enabled_engines()
    return [name for name in ENGINE_SPECS if flags.get(name, False)]


def enabled_engines() -> list[Engine]:
    engines = []
    for name in enabled_engine_names():
        engine = load_engine(name)
        if engine is not None:
            engines.append(engine)
    return engines


def engine_matrix() -> list[dict]:
    """Estado de cada engine para a UI: capabilities + disponibilidade + motivo.

    Roda no /engines e é barato de propósito — nenhum modelo é carregado aqui.
    """
    flags = get_settings().enabled_engines()
    matrix = []

    for name, (module_path, class_name) in ENGINE_SPECS.items():
        enabled = flags.get(name, False)
        engine = load_engine(name)

        if engine is None:
            matrix.append(
                {
                    "name": name,
                    "label": name,
                    "enabled": enabled,
                    "implemented": False,
                    "available": False,
                    "reason": "ainda não implementada",
                    "remedy": None,
                    "capabilities": None,
                }
            )
            continue

        try:
            av = engine.check_availability()
        except Exception as exc:  # check_availability não deveria levantar; se levantar, é dado
            log.warning("check_availability de %s levantou: %s", name, exc)
            av = Availability(available=False, reason=f"falha ao verificar: {exc}")

        matrix.append(
            {
                "name": name,
                "label": engine.label,
                "enabled": enabled,
                "implemented": True,
                "available": av.available,
                "reason": av.reason,
                "remedy": av.remedy,
                "model_version": av.model_version,
                "capabilities": vars(engine.capabilities),
            }
        )

    return matrix


def unload_all() -> None:
    for engine in _instances.values():
        try:
            engine.unload()
        except Exception as exc:
            log.warning("unload de %s falhou: %s", engine.name, exc)
