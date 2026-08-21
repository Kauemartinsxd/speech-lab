"""
Transcrição FIEL (literal) de fala em português.

Usa um modelo wav2vec2 CTC (jonatasgrosman/wav2vec2-large-xlsr-53-portuguese)
com decodificação greedy e SEM modelo de linguagem. Diferente do Whisper, o CTC
puro não "conserta" a fala: se a pessoa disser "probrema", ele tende a escrever
"probrema" — exatamente o que queremos para avaliar qualidade de fala/leitura.
"""
from __future__ import annotations

import io
import os
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

MODEL_ID = os.environ.get("ASR_MODEL", "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese")
TARGET_SR = 16000


@dataclass
class Transcription:
    text: str
    chars: list[dict]  # [{"c": "a", "conf": 0.98}, ...] confiança por caractere emitido
    duration_s: float
    infer_s: float
    device: str
    model: str


class LiteralASR:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self._lock = threading.Lock()
        self.model = None
        self.processor = None
        self.device = device
        self.loaded = False
        self.load_error: str | None = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        t0 = time.time()
        if self.device is None:
            env = os.environ.get("ASR_DEVICE", "").strip().lower()
            if env:
                self.device = env
            elif torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        print(f"[asr] carregando {self.model_id} em {self.device} ...", flush=True)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        model.eval()
        try:
            model.to(self.device)
        except Exception as e:  # pragma: no cover
            print(f"[asr] falha ao mover para {self.device} ({e}); usando cpu", flush=True)
            self.device = "cpu"
            model.to("cpu")
        self.model = model
        self.loaded = True
        print(f"[asr] pronto em {time.time() - t0:.1f}s", flush=True)

    # ------------------------------------------------------------- inference
    def transcribe(self, audio: np.ndarray, sr: int) -> Transcription:
        import torch

        if not self.loaded:
            raise RuntimeError("modelo ainda não carregado")

        audio = to_mono_float32(audio)
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)
            sr = TARGET_SR

        # normalização de amplitude leve (evita gravação muito baixa)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / max(peak, 0.1) * 0.9

        duration = len(audio) / sr
        t0 = time.time()
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)

        with self._lock, torch.inference_mode():
            logits = self.model(input_values).logits[0]  # (T, V)
            probs = torch.softmax(logits.float(), dim=-1).cpu()

        ids = probs.argmax(-1)
        confs = probs.max(-1).values

        # decodificação greedy CTC manual, para termos confiança por caractere
        vocab = self.processor.tokenizer.convert_ids_to_tokens(list(range(probs.shape[-1])))
        pad_id = self.processor.tokenizer.pad_token_id
        delim = self.processor.tokenizer.word_delimiter_token or "|"

        chars: list[dict] = []
        prev = None
        for i, tid in enumerate(ids.tolist()):
            if tid == pad_id:
                prev = tid
                continue
            if tid == prev:
                # repetição do mesmo token: mantém a maior confiança
                if chars:
                    chars[-1]["conf"] = max(chars[-1]["conf"], float(confs[i]))
                continue
            tok = vocab[tid]
            if tok in ("<s>", "</s>", "<unk>"):
                prev = tid
                continue
            c = " " if tok == delim else tok
            chars.append({"c": c, "conf": float(confs[i])})
            prev = tid

        text = "".join(ch["c"] for ch in chars)
        text = " ".join(text.split())
        # remove espaços iniciais/finais também da lista de chars
        while chars and chars[0]["c"] == " ":
            chars.pop(0)
        while chars and chars[-1]["c"] == " ":
            chars.pop()

        return Transcription(
            text=text,
            chars=chars,
            duration_s=duration,
            infer_s=time.time() - t0,
            device=self.device,
            model=self.model_id,
        )


# --------------------------------------------------------------- audio utils
def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    return audio


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(audio).unsqueeze(0)
    t = AF.resample(t, sr_in, sr_out)
    return t.squeeze(0).numpy()


def decode_audio_bytes(data: bytes, filename: str = "audio") -> tuple[np.ndarray, int]:
    """Decodifica WAV/FLAC/OGG via soundfile; qualquer outra coisa via ffmpeg."""
    import soundfile as sf

    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return to_mono_float32(audio), int(sr)
    except Exception:
        pass

    # fallback: ffmpeg -> wav 16k mono
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
        "-f", "wav", "-ac", "1", "-ar", str(TARGET_SR), "-acodec", "pcm_s16le", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError("Formato de áudio não suportado e ffmpeg não encontrado") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg falhou ao decodificar '{filename}': {e.stderr.decode(errors='ignore')[:300]}") from e
    audio, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
    return to_mono_float32(audio), int(sr)
