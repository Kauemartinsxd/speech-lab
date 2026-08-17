"""Interface comum das engines de análise de fala.

Este módulo NÃO importa torch, transformers nem qualquer SDK pesado: o registry
precisa importá-lo barato para montar a matriz de engines antes de qualquer
execução. Cada engine concreta faz seus imports pesados dentro de `run()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar


class EngineStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"  # falta modelo/binário/chave — previsível
    ERROR = "error"  # rodou e quebrou — inesperado
    TIMEOUT = "timeout"


class DeviationType(str, Enum):
    """O que aconteceu com o segmento, mecanicamente."""

    SUBSTITUICAO = "substituicao"
    ELISAO = "elisao"
    EPENTESE = "epentese"
    METATESE = "metatese"


class DeviationCategory(str, Enum):
    """Como o desvio deve ser lido. Nunca use a palavra "erro" para nenhum destes."""

    DIVERGENCIA_DE_DECODIFICACAO = "divergencia_de_decodificacao"
    VARIANTE_VERNACULAR = "variante_vernacular"
    DESVIO_ATIPICO = "desvio_atipico"


@dataclass(frozen=True)
class EngineCapabilities:
    """Declarativo. A UI lê isto — nada de regra por engine hardcoded no front."""

    produces_orthographic: bool
    produces_phonemic: bool
    produces_word_timings: bool
    produces_phone_timings: bool
    produces_confidence: bool
    requires_reference_text: bool
    sends_audio_externally: bool
    """LGPD: dispara o aviso na UI e exige opt-in explícito no .env."""
    is_reproducible: bool
    """False no gemini_audio: opina, não mede. A UI marca como qualitativo."""
    normalizes_to_standard: bool
    """True no whisper: "conserta" a fala. A UI marca como controle negativo."""


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None
    """Ex.: "modelo não baixado", "AZURE_SPEECH_KEY ausente"."""
    remedy: str | None = None
    """Comando ou passo que resolve. Aparece na matriz de engines."""
    model_version: str | None = None


@dataclass(frozen=True)
class AudioInput:
    path: Path
    """WAV mono 16 kHz PCM s16le, já normalizado. Toda engine recebe o mesmo sinal."""
    sample_rate: int
    duration_s: float
    sha256: str


@dataclass(frozen=True)
class EngineRequest:
    audio: AudioInput
    reference_text: str | None
    options: dict[str, Any] = field(default_factory=dict)
    """Overrides vindos da UI em runtime (ex.: threshold de GOP). Nada hardcoded."""


@dataclass
class Token:
    """Unidade genérica: palavra OU fonema, conforme o campo que a carrega."""

    text: str
    """Grafema ou símbolo IPA, já normalizado pelo inventory."""
    start_s: float | None = None
    end_s: float | None = None
    confidence: float | None = None
    """0..1. None quando a engine não produz confiança reproduzível."""
    raw_text: str | None = None
    """Símbolo original antes da normalização, para auditoria."""


@dataclass
class GopScore:
    phoneme: str
    index: int
    start_s: float
    end_s: float
    gop_posterior: float
    """Média do log-posterior do fonema esperado no intervalo alinhado."""
    gop_ratio: float
    """log P(esperado) - max_q log P(q). Mais discriminativo que o posterior puro."""
    below_threshold: bool
    """Avaliado contra o threshold vindo de `EngineRequest.options`, não de constante."""


@dataclass
class Deviation:
    expected: str | None
    """None => epêntese (apareceu algo que não estava previsto)."""
    observed: str | None
    """None => elisão (o previsto não apareceu)."""
    index_canonical: int | None
    index_observed: int | None
    type: DeviationType
    category: DeviationCategory
    process: str | None = None
    """Id do processo fonológico que casou: "metatese", "rotacismo", ..."""
    word: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    gop: float | None = None
    evidence: str | None = None
    """Qual regra do variantes_ptbr.yaml casou. Sem isto a classificação é opaca."""


@dataclass
class EngineResult:
    engine: str
    status: EngineStatus
    unavailable_reason: str | None = None
    transcript: str | None = None
    """Ortográfico CRU. Nenhum pós-processamento, nenhuma correção."""
    words: list[Token] = field(default_factory=list)
    phonemes: list[Token] = field(default_factory=list)
    """IPA efetivamente reconhecido, sem restrição ao texto de referência."""
    canonical_phonemes: list[Token] = field(default_factory=list)
    """IPA esperado, derivado do texto de referência via G2P."""
    gop: list[GopScore] = field(default_factory=list)
    deviations: list[Deviation] = field(default_factory=list)
    """Fica VAZIO na engine. Preenchido pela camada de classificação, depois."""
    metrics: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    """Payload bruto, para conferir qualquer número depois sem re-executar."""
    latency_ms: int = 0
    cost_estimate_usd: float | None = None
    model_version: str | None = None
    device: str | None = None


class Engine(ABC):
    name: ClassVar[str]
    label: ClassVar[str]
    capabilities: ClassVar[EngineCapabilities]

    @abstractmethod
    def check_availability(self) -> Availability:
        """Barato e sem efeito colateral: NÃO carrega modelo, NÃO baixa peso, NÃO
        chama a rede. Roda no /engines para a UI desenhar a matriz antes de
        qualquer execução. Nunca levanta exceção."""

    @abstractmethod
    async def run(self, req: EngineRequest) -> EngineResult:
        """Nunca levanta exceção: devolve EngineResult com status ERROR.

        É isto que faz a degradação graciosa ser real — o runner não depende de
        try/except espalhado para manter as outras engines de pé.
        """

    def unload(self) -> None:
        """Libera os pesos da memória. Default no-op. Obrigatório em 16 GB."""
        return None

    def estimate_cost(self, req: EngineRequest) -> float | None:
        """Tarifa vem do .env — preço muda, não entra em código."""
        return None


def audio_missing_result(engine: str, req: EngineRequest) -> EngineResult | None:
    """Guarda comum, chamada antes de carregar qualquer modelo.

    Sem áudio no disco não há o que medir, e carregar 1,5 GB de pesos para
    então falhar na leitura do arquivo é desperdício puro — em 16 GB, é também
    memória que faltaria para as outras engines.
    """
    if not req.audio.path.exists():
        return EngineResult(
            engine=engine,
            status=EngineStatus.ERROR,
            unavailable_reason=f"áudio não encontrado: {req.audio.path}",
        )
    return None


def unavailable_result(engine: str, av: Availability) -> EngineResult:
    """Atalho para o caso mais comum: a engine não pode rodar e diz por quê."""
    reason = av.reason or "indisponível"
    if av.remedy:
        reason = f"{reason} — {av.remedy}"
    return EngineResult(
        engine=engine,
        status=EngineStatus.UNAVAILABLE,
        unavailable_reason=reason,
        model_version=av.model_version,
    )
