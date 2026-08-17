from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def tone_wav(tmp_path_factory) -> Path:
    """Um WAV real gerado pelo ffmpeg, fora do formato canônico de propósito:
    44.1 kHz estéreo, para que a normalização tenha o que fazer."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg ausente")

    dest = tmp_path_factory.mktemp("audio") / "tone_44k_stereo.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.5:sample_rate=44100",
            "-ac",
            "2",
            str(dest),
        ],
        check=True,
    )
    return dest


@pytest.fixture(scope="session")
def fala_wav(tmp_path_factory) -> Path:
    """Áudio de fala de verdade via `say`, para testes que precisam de sinal
    com estrutura fonética. Pulado fora do macOS."""
    if sys.platform != "darwin":
        pytest.skip("`say` só existe no macOS")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg ausente")

    workdir = tmp_path_factory.mktemp("fala")
    aiff = workdir / "fala.aiff"
    subprocess.run(
        ["say", "-v", "Luciana", "-o", str(aiff), "O pobrema da conta é difícil."],
        check=True,
    )
    return aiff
