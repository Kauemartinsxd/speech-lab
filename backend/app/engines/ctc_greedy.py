"""ctc_greedy — Caminho 1.

wav2vec2 XLSR fine-tunado em pt-BR, decodificação GREEDY: argmax quadro a
quadro, colapso de CTC, e nada mais.

Sem KenLM. Sem beam search. Sem shallow fusion. Isso é requisito, não detalhe de
implementação: qualquer modelo de linguagem no caminho de decodificação
reintroduziria exatamente o viés que o lab quer medir, e "pobrema" voltaria a
sair como "problema".

Por isso o processor é carregado como `Wav2Vec2Processor` explicitamente, e não
via `AutoProcessor`: se o repositório do modelo trouxer um `language_model/`,
o Auto devolveria um `Wav2Vec2ProcessorWithLM`. Aqui o argmax é feito à mão, de
qualquer forma.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, ClassVar

from app.audio.normalize import load_waveform
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
from app.runtime.device import pick_device
from app.runtime.model_cache import get_model_cache

log = logging.getLogger(__name__)

WORD_DELIMITER = "|"


def _load_ctc(model_id: str, device: str):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return processor, model


class CtcGreedyEngine(Engine):
    name: ClassVar[str] = "ctc_greedy"
    label: ClassVar[str] = "CTC greedy (wav2vec2 XLSR pt-BR)"
    capabilities: ClassVar[EngineCapabilities] = EngineCapabilities(
        produces_orthographic=True,
        produces_phonemic=False,
        produces_word_timings=True,
        produces_phone_timings=False,
        produces_confidence=True,
        requires_reference_text=False,
        sends_audio_externally=False,
        is_reproducible=True,
        normalizes_to_standard=False,
    )

    def _model_id(self, req: EngineRequest | None = None) -> str:
        if req is not None:
            override = req.options.get("ctc_model")
            if override:
                return str(override)
        return get_settings().ctc_model

    def check_availability(self) -> Availability:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return Availability(
                available=False,
                reason=f"dependência ausente: {exc.name}",
                remedy="uv sync",
            )

        from app.engines.whisper_baseline import _hf_cache_has

        model_id = self._model_id()
        cached = _hf_cache_has(model_id)
        return Availability(
            available=True,
            reason=None if cached else "modelo ainda não baixado (~1.2 GB no primeiro uso)",
            model_version=model_id,
        )

    def unload(self) -> None:
        cache = get_model_cache()
        for key in list(cache.resident()):
            if key.startswith("ctc:"):
                cache.unload(key)

    async def run(self, req: EngineRequest) -> EngineResult:
        started = time.perf_counter()
        try:
            return await asyncio.to_thread(self._run_sync, req, started)
        except Exception as exc:
            log.exception("ctc_greedy falhou")
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

        import numpy as np
        import torch

        model_id = self._model_id(req)
        device = pick_device()

        processor, model = get_model_cache().get_or_load(
            f"ctc:{model_id}:{device}",
            lambda: _load_ctc(model_id, device),
        )

        waveform, sample_rate = load_waveform(req.audio.path)
        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

        used_device = device
        with torch.inference_mode():
            try:
                logits = model(inputs.input_values.to(device)).logits
            except (RuntimeError, NotImplementedError) as exc:
                # Alguns operadores ainda caem no MPS. Degrada para CPU em vez
                # de derrubar a engine — e registra qual device produziu o número.
                log.warning("forward em %s falhou (%s); repetindo em CPU", device, exc)
                model.to("cpu")
                logits = model(inputs.input_values).logits
                used_device = "cpu"

            probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()

        pred_ids = probs.argmax(axis=-1)
        n_frames = probs.shape[0]
        frame_s = req.audio.duration_s / n_frames if n_frames else 0.0

        blank_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
        id_to_token = {v: k for k, v in processor.tokenizer.get_vocab().items()}

        chars, char_runs = self._ctc_collapse(
            pred_ids=pred_ids,
            probs=probs,
            blank_id=blank_id,
            id_to_token=id_to_token,
            frame_s=frame_s,
        )
        words = self._group_words(chars)

        transcript = "".join(" " if c.text == WORD_DELIMITER else c.text for c in chars).strip()

        char_conf = [c.confidence for c in chars if c.confidence is not None]
        non_blank = pred_ids != blank_id

        return EngineResult(
            engine=self.name,
            status=EngineStatus.OK,
            transcript=transcript,
            words=words,
            metrics={
                # proxy de inteligibilidade: quanto o modelo "hesitou" por caractere
                "confianca_media_caractere": float(np.mean(char_conf)) if char_conf else 0.0,
                "confianca_minima_caractere": float(np.min(char_conf)) if char_conf else 0.0,
                "confianca_media_quadro": float(np.mean(probs.max(axis=-1))),
                "razao_quadros_nao_brancos": float(non_blank.mean()) if n_frames else 0.0,
                "n_palavras": float(len(words)),
                "n_quadros": float(n_frames),
            },
            raw={
                "model_id": model_id,
                "decodificacao": "greedy argmax, sem LM, sem beam search",
                "frame_s": frame_s,
                "blank_id": int(blank_id),
                # matriz de logits reduzida: 1 linha por caractere emitido
                "caracteres": [
                    {
                        "char": c.text,
                        "start_s": c.start_s,
                        "end_s": c.end_s,
                        "confianca": c.confidence,
                        "n_quadros": runs,
                    }
                    for c, runs in zip(chars, char_runs, strict=True)
                ],
            },
            latency_ms=int((time.perf_counter() - started) * 1000),
            model_version=model_id,
            device=used_device,
        )

    @staticmethod
    def _ctc_collapse(
        pred_ids: Any,
        probs: Any,
        blank_id: int,
        id_to_token: dict[int, str],
        frame_s: float,
    ) -> tuple[list[Token], list[int]]:
        """Colapso de CTC: junta quadros repetidos, descarta o branco.

        A confiança de cada caractere é a média da probabilidade daquele id nos
        quadros que o produziram — não o máximo, para que um caractere sustentado
        por muitos quadros incertos não pareça confiante.
        """
        import numpy as np

        chars: list[Token] = []
        runs: list[int] = []

        current_id: int | None = None
        run_start = 0

        def flush(end_frame: int) -> None:
            if current_id is None or current_id == blank_id:
                return
            token = id_to_token.get(int(current_id), "")
            if not token or token.startswith("<") or token.startswith("["):
                return  # tokens especiais (<pad>, <s>, [UNK]) não são fala
            frame_probs = probs[run_start:end_frame, current_id]
            chars.append(
                Token(
                    text=token,
                    start_s=round(run_start * frame_s, 4),
                    end_s=round(end_frame * frame_s, 4),
                    confidence=float(np.mean(frame_probs)),
                )
            )
            runs.append(end_frame - run_start)

        for i, pid in enumerate(pred_ids):
            if current_id is None or pid != current_id:
                flush(i)
                current_id = int(pid)
                run_start = i
        flush(len(pred_ids))

        return chars, runs

    @staticmethod
    def _group_words(chars: list[Token]) -> list[Token]:
        """Agrupa caracteres em palavras usando o delimitador do vocabulário."""
        words: list[Token] = []
        buffer: list[Token] = []

        def flush() -> None:
            if not buffer:
                return
            confs = [c.confidence for c in buffer if c.confidence is not None]
            words.append(
                Token(
                    text="".join(c.text for c in buffer),
                    start_s=buffer[0].start_s,
                    end_s=buffer[-1].end_s,
                    confidence=(sum(confs) / len(confs)) if confs else None,
                )
            )
            buffer.clear()

        for char in chars:
            if char.text == WORD_DELIMITER:
                flush()
            else:
                buffer.append(char)
        flush()

        return words
