"""Gera as amostras de exemplo com o TTS do macOS.

O `say` articula a string literalmente: pedir "pobrema" produz áudio de
"pobrema", não de "problema". Isso dá um sinal real para provar o comportamento
do decoder do Whisper hoje, sem depender de gravação.

Limite honesto: é voz ADULTA SINTÉTICA. Toda amostra nasce com sintetico=True e
fica fora das métricas de fala infantil — serve para validar o pipeline, não
para responder à pergunta de pesquisa.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

import yaml  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

from app.audio.normalize import normalize_audio  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import get_engine, init_db  # noqa: E402
from app.models import Annotation, Sample, SampleSource  # noqa: E402

SEED_FILE = REPO_ROOT / "data" / "seed" / "seed_pairs.yaml"


def say_to_aiff(text: str, voice: str, dest: Path) -> None:
    proc = subprocess.run(
        ["say", "-v", voice, "-o", str(dest), text],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"say falhou: {proc.stderr.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gera as amostras de exemplo via TTS")
    parser.add_argument("--voice", default=None, help="voz do say (default: a do YAML)")
    parser.add_argument("--only", default=None, help="gera apenas o par com este id")
    parser.add_argument("--force", action="store_true", help="regera mesmo se já existir")
    args = parser.parse_args()

    if sys.platform != "darwin":
        print("erro: este seed usa o `say` do macOS", file=sys.stderr)
        return 1

    spec = yaml.safe_load(SEED_FILE.read_text(encoding="utf-8"))
    voice = args.voice or spec.get("voz_padrao", "Luciana")
    pares = spec["pares"]
    if args.only:
        pares = [p for p in pares if p["id"] == args.only]
        if not pares:
            print(f"erro: par '{args.only}' não existe em {SEED_FILE.name}", file=sys.stderr)
            return 1

    settings = get_settings()
    settings.abs_audio_dir.mkdir(parents=True, exist_ok=True)
    init_db()

    criados = 0
    pulados = 0

    with Session(get_engine()) as session:
        for par in pares:
            label = f"seed:{par['id']}"

            existente = session.exec(select(Sample).where(Sample.label == label)).first()
            if existente is not None and not args.force:
                pulados += 1
                continue

            with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
                aiff = Path(tmp.name)
            try:
                say_to_aiff(par["falado"], voice, aiff)
                staged = settings.abs_audio_dir / f"seed-{par['id']}.wav"
                normalized = normalize_audio(aiff, staged)
                final_path = settings.abs_audio_dir / f"{normalized.sha256[:16]}.wav"
                if final_path.exists():
                    staged.unlink(missing_ok=True)
                else:
                    staged.rename(final_path)
            finally:
                aiff.unlink(missing_ok=True)

            sample = existente or Sample(sha256=normalized.sha256)
            sample.label = label
            sample.source = SampleSource.SEED_TTS
            sample.original_filename = None
            sample.audio_path = str(final_path)
            sample.sha256 = normalized.sha256
            sample.duration_s = normalized.duration_s
            sample.sample_rate = normalized.sample_rate
            sample.reference_text = par["referencia"]
            sample.sintetico = True
            sample.tts_voice = voice
            sample.notes = f"processo: {par['processo']}"
            session.add(sample)
            session.commit()
            session.refresh(sample)

            # O ground truth já é conhecido: sabemos exatamente o que a voz falou.
            anotacao = session.exec(
                select(Annotation).where(Annotation.sample_id == sample.id)
            ).first()
            if anotacao is None:
                anotacao = Annotation(sample_id=sample.id, transcript_fiel=par["falado"])
            anotacao.transcript_fiel = par["falado"]
            anotacao.annotator = "seed_tts"
            anotacao.deviations = [
                {
                    "expected": par["palavra_padrao"],
                    "observed": par["palavra_variante"],
                    "category": "variante_vernacular",
                    "process": par["processo"],
                }
            ]
            session.add(anotacao)
            session.commit()

            print(
                f"  {par['id']:32} {normalized.duration_s:5.2f}s  "
                f"{par['palavra_padrao']} -> {par['palavra_variante']}"
            )
            criados += 1

    print(f"\n{criados} amostra(s) gerada(s), {pulados} já existia(m). Voz: {voice}")
    if pulados:
        print("use --force para regerar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
