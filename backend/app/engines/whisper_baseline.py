"""whisper_baseline — controle negativo.

Esta engine existe para DEMONSTRAR o problema, não para resolvê-lo.

O Whisper decodifica autorregressivamente: cada token é condicionado nos
anteriores, o que embute um modelo de linguagem. "pobrema" tem probabilidade
baixíssima em pt-BR e "problema" tem alta — o decoder escolhe a alta e o desvio
que o lab quer medir desaparece na saída.

`capabilities.normalizes_to_standard = True` faz a UI marcar isso explicitamente.

Duas entradas compartilham o mesmo modelo em cache:
  - whisper_baseline: parâmetros padrão da biblioteca;
  - whisper_strict:   condition_on_previous_text=False e temperature=0, para
                      verificar se tirar o condicionamento no texto anterior
                      muda alguma coisa.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, ClassVar

from app.config import get_settings
from app.engines.base import (
    Availability,
    Engine,
    EngineCapabilities,
    EngineRequest,
    EngineResult,
    EngineStatus,
    Token,
    audio_missing_result,
)
from app.runtime.model_cache import get_model_cache

log = logging.getLogger(__name__)

WHISPER_CAPS = EngineCapabilities(
    produces_orthographic=True,
    produces_phonemic=False,
    produces_word_timings=True,
    produces_phone_timings=False,
    produces_confidence=True,
    requires_reference_text=False,
    sends_audio_externally=False,
    is_reproducible=True,
    normalizes_to_standard=True,
)


def _hf_cache_has(repo_id: str) -> bool:
    """Checagem barata de presença no cache do HuggingFace. Não baixa nada."""
    import os

    cache_root = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    hub = cache_root / "hub" if (cache_root / "hub").exists() else cache_root
    folder = "models--" + repo_id.replace("/", "--")
    return (hub / folder).exists()


def _load_whisper(model_name: str, compute_type: str):
    from faster_whisper import WhisperModel

    # CTranslate2 não tem backend Metal: em Apple Silicon isto é sempre CPU.
    return WhisperModel(model_name, device="cpu", compute_type=compute_type)


class _WhisperBase(Engine):
    capabilities: ClassVar[EngineCapabilities] = WHISPER_CAPS
    transcribe_kwargs: ClassVar[dict[str, Any]] = {}

    def _cache_key(self) -> str:
        s = get_settings()
        return f"whisper:{s.whisper_model}:{s.whisper_compute_type}"

    def check_availability(self) -> Availability:
        settings = get_settings()
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return Availability(
                available=False,
                reason="faster-whisper não instalado",
                remedy="uv sync",
            )

        repo = f"Systran/faster-whisper-{settings.whisper_model}"
        cached = _hf_cache_has(repo)
        return Availability(
            available=True,
            reason=None if cached else "modelo ainda não baixado (~1.5 GB no primeiro uso)",
            model_version=settings.whisper_model,
        )

    async def run(self, req: EngineRequest) -> EngineResult:
        started = time.perf_counter()
        try:
            return await asyncio.to_thread(self._run_sync, req, started)
        except Exception as exc:
            log.exception("%s falhou", self.name)
            return EngineResult(
                engine=self.name,
                status=EngineStatus.ERROR,
                unavailable_reason=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

    def _run_sync(self, req: EngineRequest, started: float) -> EngineResult:
        faltando = audio_missing_result(self.name, req)
        if faltando is not None:
            return faltando

        settings = get_settings()
        model = get_model_cache().get_or_load(
            self._cache_key(),
            lambda: _load_whisper(settings.whisper_model, settings.whisper_compute_type),
        )

        segments, info = model.transcribe(
            str(req.audio.path),
            language="pt",
            word_timestamps=True,
            **self.transcribe_kwargs,
        )

        words: list[Token] = []
        pieces: list[str] = []
        seg_dump: list[dict[str, Any]] = []

        for seg in segments:  # gerador: só executa de verdade ao ser consumido
            pieces.append(seg.text)
            seg_dump.append(
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "avg_logprob": seg.avg_logprob,
                    "no_speech_prob": seg.no_speech_prob,
                    "temperature": getattr(seg, "temperature", None),
                }
            )
            for w in seg.words or []:
                words.append(
                    Token(
                        text=w.word.strip(),
                        start_s=w.start,
                        end_s=w.end,
                        confidence=w.probability,
                    )
                )

        transcript = "".join(pieces).strip()
        confidences = [w.confidence for w in words if w.confidence is not None]

        return EngineResult(
            engine=self.name,
            status=EngineStatus.OK,
            transcript=transcript,
            words=words,
            metrics={
                "confianca_media": (sum(confidences) / len(confidences)) if confidences else 0.0,
                "n_palavras": float(len(words)),
                "language_probability": float(info.language_probability),
            },
            raw={
                "segments": seg_dump,
                "language": info.language,
                "duration": info.duration,
                "transcribe_kwargs": {k: str(v) for k, v in self.transcribe_kwargs.items()},
            },
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_version=settings.whisper_model,
            device="cpu",
        )


class WhisperBaselineEngine(_WhisperBase):
    name: ClassVar[str] = "whisper_baseline"
    label: ClassVar[str] = "Whisper (controle negativo)"
    transcribe_kwargs: ClassVar[dict[str, Any]] = {}


class WhisperStrictEngine(_WhisperBase):
    name: ClassVar[str] = "whisper_strict"
    label: ClassVar[str] = "Whisper sem condicionamento"
    transcribe_kwargs: ClassVar[dict[str, Any]] = {
        # sem histórico como prompt e sem fallback de temperatura: se ainda
        # assim normalizar, a normalização não vem do condicionamento no texto
        # anterior — vem do próprio decoder.
        "condition_on_previous_text": False,
        "temperature": 0.0,
    }
