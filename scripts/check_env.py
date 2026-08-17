"""Diagnóstico do ambiente: o que está pronto e o que falta.

Roda com `make check`. Nunca falha por falta de componente opcional — diz o que
está faltando e qual comando resolve.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

OK = "\033[32m ok \033[0m"
FALTA = "\033[33mfalta\033[0m"
ERRO = "\033[31merro \033[0m"


def linha(status: str, nome: str, detalhe: str = "") -> None:
    print(f"  [{status}] {nome:34} {detalhe}")


def check_binarios() -> bool:
    print("\nbinários do sistema")
    tudo_ok = True
    for binario, remedio in (
        ("ffmpeg", "brew install ffmpeg"),
        ("ffprobe", "brew install ffmpeg"),
        ("espeak-ng", "brew install espeak-ng"),
        ("say", "só existe no macOS; o seed por TTS depende dele"),
    ):
        caminho = shutil.which(binario)
        if caminho:
            linha(OK, binario, caminho)
        else:
            linha(FALTA, binario, remedio)
            if binario in ("ffmpeg", "ffprobe"):
                tudo_ok = False
    return tudo_ok


def check_espeak_lib() -> None:
    print("\nlib do espeak (phonemizer)")
    configurado = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
    candidatos = [configurado] if configurado else []
    candidatos += [
        "/opt/homebrew/lib/libespeak-ng.dylib",
        "/usr/local/lib/libespeak-ng.dylib",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so",
    ]
    for c in candidatos:
        if c and Path(c).exists():
            origem = "via PHONEMIZER_ESPEAK_LIBRARY" if c == configurado else "encontrada"
            linha(OK, "libespeak-ng", f"{c} ({origem})")
            return
    linha(
        FALTA,
        "libespeak-ng",
        "defina PHONEMIZER_ESPEAK_LIBRARY no .env (macOS arm64 não acha sozinho)",
    )


def check_python() -> None:
    print("\npacotes python")
    for modulo, remedio in (
        ("torch", "uv sync"),
        ("torchaudio", "uv sync"),
        ("transformers", "uv sync"),
        ("faster_whisper", "uv sync"),
        ("soundfile", "uv sync"),
        ("phonemizer", "uv sync --extra phonemes"),
        ("azure.cognitiveservices.speech", "uv sync --extra azure"),
        ("google.genai", "uv sync --extra gemini"),
    ):
        if importlib.util.find_spec(modulo) is not None:
            linha(OK, modulo)
        else:
            linha(FALTA, modulo, remedio)


def check_device() -> None:
    print("\ndevice")
    try:
        from app.runtime import device

        info = device.describe()
        linha(OK, "inferência", str(info["inference_device"]))
        linha(OK, "alinhamento/GOP", f"{info['alignment_device']}  (determinístico)")
        linha(OK, "torch", str(info.get("torch_version")))
        if info.get("mps_available") and info["inference_device"] == "mps":
            linha(
                OK,
                "nota",
                "faster-whisper roda em CPU: CTranslate2 não tem backend Metal",
            )
    except Exception as exc:
        linha(ERRO, "device", str(exc))


def check_modelos() -> None:
    print("\nmodelos no cache do HuggingFace")
    from app.config import get_settings
    from app.engines.whisper_baseline import _hf_cache_has

    settings = get_settings()
    for repo, rotulo in (
        (f"Systran/faster-whisper-{settings.whisper_model}", "whisper"),
        (settings.ctc_model, "ctc_greedy"),
        (settings.phoneme_model, "phoneme_gop"),
    ):
        if _hf_cache_has(repo):
            linha(OK, rotulo, repo)
        else:
            linha(FALTA, rotulo, f"{repo} (baixa no primeiro uso)")


def check_engines() -> None:
    print("\nengines")
    from app.engines.registry import engine_matrix

    for e in engine_matrix():
        if not e["implemented"]:
            linha(FALTA, e["name"], "ainda não implementada")
            continue
        habilitada = "habilitada" if e["enabled"] else "desabilitada no .env"
        detalhe = f"{habilitada}"
        if e.get("reason"):
            detalhe += f" — {e['reason']}"
        if e["capabilities"] and e["capabilities"]["sends_audio_externally"]:
            detalhe += "  [ENVIA ÁUDIO PARA FORA]"
        linha(OK if e["available"] else FALTA, e["name"], detalhe)


def check_chaves() -> None:
    print("\nchaves de API (engines externas)")
    for var in ("AZURE_SPEECH_KEY", "GEMINI_API_KEY"):
        linha(OK if os.environ.get(var) else FALTA, var, "opt-in explícito por engine")


def check_privacidade() -> None:
    print("\nprivacidade")
    from app.config import get_settings

    s = get_settings()
    linha(
        OK,
        "descarte de áudio",
        "LIGADO (quebra player e re-execução)"
        if s.discard_audio_after_features
        else "desligado",
    )
    externas = [
        n for n, on in s.enabled_engines().items() if on and n in ("azure_pa", "gemini_audio")
    ]
    linha(
        OK,
        "engines que enviam áudio",
        ", ".join(externas) if externas else "nenhuma habilitada",
    )


def main() -> int:
    print("speech-lab — diagnóstico do ambiente")
    essencial_ok = check_binarios()
    check_espeak_lib()
    check_python()
    check_device()
    check_modelos()
    check_engines()
    check_chaves()
    check_privacidade()

    print()
    if not essencial_ok:
        print("ffmpeg é obrigatório: brew install ffmpeg")
        return 1
    print("essenciais presentes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
