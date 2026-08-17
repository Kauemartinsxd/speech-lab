"""Normalização de áudio.

Se este módulo estiver errado, todas as engines recebem sinais diferentes e o
comparativo deixa de comparar coisa alguma.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf
from app.audio.normalize import (
    TARGET_SAMPLE_RATE,
    AudioNormalizationError,
    load_waveform,
    normalize_audio,
    sha256_of,
)


def test_converte_para_mono_16k_pcm(tone_wav: Path, tmp_path: Path):
    dest = tmp_path / "out.wav"
    result = normalize_audio(tone_wav, dest)

    info = sf.info(str(dest))
    assert info.samplerate == TARGET_SAMPLE_RATE
    assert info.channels == 1
    assert info.subtype == "PCM_16"

    assert result.sample_rate == TARGET_SAMPLE_RATE
    assert result.duration_s == pytest.approx(1.5, abs=0.05)
    assert len(result.sha256) == 64


def test_e_deterministica(tone_wav: Path, tmp_path: Path):
    """Duas normalizações do mesmo arquivo têm que dar o mesmo sha256 — é o que
    permite deduplicar amostras e confiar que uma re-execução vê o mesmo sinal."""
    a = normalize_audio(tone_wav, tmp_path / "a.wav")
    b = normalize_audio(tone_wav, tmp_path / "b.wav")
    assert a.sha256 == b.sha256


def test_aceita_formato_nao_wav(fala_wav: Path, tmp_path: Path):
    """A entrada real vem do MediaRecorder (webm/ogg) ou do `say` (aiff)."""
    result = normalize_audio(fala_wav, tmp_path / "fala.wav")
    assert result.duration_s > 0.5
    assert sf.info(str(result.path)).samplerate == TARGET_SAMPLE_RATE


def test_waveform_carrega_mono_normalizado(tone_wav: Path, tmp_path: Path):
    result = normalize_audio(tone_wav, tmp_path / "out.wav")
    data, sr = load_waveform(result.path)

    assert sr == TARGET_SAMPLE_RATE
    assert data.ndim == 1
    assert len(data) == pytest.approx(TARGET_SAMPLE_RATE * 1.5, rel=0.05)
    assert data.max() <= 1.0 and data.min() >= -1.0


def test_arquivo_inexistente_levanta(tmp_path: Path):
    with pytest.raises(AudioNormalizationError, match="não existe"):
        normalize_audio(tmp_path / "fantasma.wav", tmp_path / "out.wav")


def test_arquivo_vazio_levanta(tmp_path: Path):
    vazio = tmp_path / "vazio.wav"
    vazio.write_bytes(b"")
    with pytest.raises(AudioNormalizationError, match="vazio"):
        normalize_audio(vazio, tmp_path / "out.wav")


def test_arquivo_corrompido_levanta(tmp_path: Path):
    lixo = tmp_path / "lixo.wav"
    lixo.write_bytes(b"isto definitivamente nao e audio" * 10)
    with pytest.raises(AudioNormalizationError):
        normalize_audio(lixo, tmp_path / "out.wav")


def test_sha256_muda_com_o_conteudo(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"conteudo a")
    b.write_bytes(b"conteudo b")
    assert sha256_of(a) != sha256_of(b)
