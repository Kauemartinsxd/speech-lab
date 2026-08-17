"""Valida se o TTS realmente articulou a variante, e não a forma padrão.

Sem esta checagem o seed sintético é inútil como teste: se o `say` renderiza
"bicicreta" com o mesmo áudio de "bicicleta", então uma engine que transcreve
"bicicleta" está sendo FIEL, não normalizando — e qualquer conclusão sobre
normalização fica invertida.

O método não depende de nenhum ASR: sintetiza as duas frases com a mesma voz e
mede a distância entre os sinais. Se forem praticamente iguais, quem normalizou
foi o TTS, e o par não serve como caso de teste.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import yaml  # noqa: E402
from app.audio.normalize import normalize_audio  # noqa: E402

SEED_FILE = REPO_ROOT / "data" / "seed" / "seed_pairs.yaml"

# Abaixo disto os sinais são praticamente o mesmo áudio.
LIMIAR_RMS = 0.01
LIMIAR_DURACAO_S = 0.02


def sintetizar(texto: str, voz: str, workdir: Path, nome: str) -> np.ndarray:
    aiff = workdir / f"{nome}.aiff"
    subprocess.run(["say", "-v", voz, "-o", str(aiff), texto], check=True)
    wav = normalize_audio(aiff, workdir / f"{nome}.wav")
    data, _ = sf.read(str(wav.path), dtype="float32")
    return data


def main() -> int:
    if sys.platform != "darwin":
        print("erro: precisa do `say` do macOS", file=sys.stderr)
        return 1

    spec = yaml.safe_load(SEED_FILE.read_text(encoding="utf-8"))
    voz = spec.get("voz_padrao", "Luciana")

    print(f"validando o seed sintético (voz: {voz})")
    print(f"{'par':32} {'Δdur':>8} {'RMS':>8}  veredito")
    print("-" * 72)

    invalidos = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for par in spec["pares"]:
            padrao = sintetizar(par["referencia"], voz, workdir, f"{par['id']}-p")
            variante = sintetizar(par["falado"], voz, workdir, f"{par['id']}-v")

            d_dur = abs(len(padrao) - len(variante)) / 16_000
            n = min(len(padrao), len(variante))
            rms = float(np.sqrt(np.mean((padrao[:n] - variante[:n]) ** 2)))

            distinto = d_dur > LIMIAR_DURACAO_S or rms > LIMIAR_RMS
            veredito = "articulou a variante" if distinto else "TTS NORMALIZOU — par inválido"
            if not distinto:
                invalidos.append(par["id"])

            print(f"{par['id']:32} {d_dur:7.3f}s {rms:8.4f}  {veredito}")

    print()
    if invalidos:
        print(f"{len(invalidos)} par(es) inválido(s): {', '.join(invalidos)}")
        print("nestes, transcrever a forma padrão é fidelidade, não normalização.")
        return 1

    print("todos os pares foram articulados de forma distinta da referência.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
