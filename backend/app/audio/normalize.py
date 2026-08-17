"""Normalização de áudio para WAV mono 16 kHz PCM s16le via ffmpeg.

Todas as engines recebem exatamente o mesmo sinal. Se cada uma reamostrasse por
conta própria, diferenças de resampler entrariam na medida e o comparativo
deixaria de ser comparativo.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"


class AudioNormalizationError(RuntimeError):
    pass


class FfmpegMissingError(AudioNormalizationError):
    def __init__(self) -> None:
        super().__init__("ffmpeg não encontrado no PATH — instale com: brew install ffmpeg")


@dataclass(frozen=True)
class NormalizedAudio:
    path: Path
    sample_rate: int
    channels: int
    duration_s: float
    sha256: str


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_duration_s(path: Path) -> float:
    """Duração via ffprobe, com fallback para leitura do cabeçalho WAV."""
    if ffprobe_available():
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            try:
                return float(json.loads(proc.stdout)["format"]["duration"])
            except (KeyError, ValueError, json.JSONDecodeError):
                pass

    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def normalize_audio(src: Path, dest: Path) -> NormalizedAudio:
    """Converte qualquer entrada suportada pelo ffmpeg para o formato canônico.

    Levanta AudioNormalizationError em qualquer falha — quem chama decide o que
    fazer. Este caminho roda antes das engines, então falhar alto aqui é correto:
    sem áudio normalizado não há experimento.
    """
    if not ffmpeg_available():
        raise FfmpegMissingError()

    src = Path(src)
    dest = Path(dest)

    if not src.exists():
        raise AudioNormalizationError(f"arquivo de origem não existe: {src}")
    if src.stat().st_size == 0:
        raise AudioNormalizationError(f"arquivo de origem está vazio: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-acodec",
            TARGET_CODEC,
            "-map_metadata",
            "-1",  # metadado pode carregar dado pessoal; não sobrevive à normalização
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        raise AudioNormalizationError(
            f"ffmpeg falhou ({proc.returncode}): {proc.stderr.strip() or 'sem stderr'}"
        )
    if not dest.exists() or dest.stat().st_size == 0:
        raise AudioNormalizationError("ffmpeg terminou sem erro mas não produziu áudio")

    duration = probe_duration_s(dest)
    if duration <= 0:
        raise AudioNormalizationError("áudio normalizado tem duração zero")

    return NormalizedAudio(
        path=dest,
        sample_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        duration_s=duration,
        sha256=sha256_of(dest),
    )


def load_waveform(path: Path) -> tuple["object", int]:
    """Carrega o WAV normalizado como float32 mono em [-1, 1].

    Devolve (numpy.ndarray, sample_rate). Import de soundfile é local para
    manter este módulo barato de importar.
    """
    import numpy as np
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data, int(sample_rate)
