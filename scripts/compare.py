"""Comparativo em linha de comando: roda as engines numa amostra e imprime lado a lado.

A UI faz o mesmo com mais detalhe; isto existe para reproduzir o experimento
sem navegador e para colar o resultado num caderno de laboratório.

    uv run python scripts/compare.py --label seed:metatese_problema
    uv run python scripts/compare.py --todas
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.db import get_engine, init_db  # noqa: E402
from app.engines.base import AudioInput, EngineRequest, EngineStatus  # noqa: E402
from app.engines.registry import enabled_engine_names, load_engine  # noqa: E402
from app.engines.runner import run_engines  # noqa: E402
from app.models import Annotation, Sample  # noqa: E402
from sqlmodel import Session, select  # noqa: E402

VERDE = "\033[32m"
AMARELO = "\033[33m"
CINZA = "\033[90m"
RESET = "\033[0m"


def normalizar(texto: str) -> str:
    import unicodedata

    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    return " ".join("".join(c for c in p if c.isalnum()) for p in sem_acento.split())


async def comparar(sample: Sample, verdade: str | None, engine_names: list[str]) -> None:
    audio = AudioInput(
        path=Path(sample.audio_path),
        sample_rate=sample.sample_rate,
        duration_s=sample.duration_s,
        sha256=sample.sha256,
    )
    req = EngineRequest(audio=audio, reference_text=sample.reference_text, options={})
    engines = [e for e in (load_engine(n) for n in engine_names) if e is not None]

    print(f"\n{'=' * 78}")
    print(f"amostra    {sample.label}   {sample.duration_s:.2f}s")
    if sample.reference_text:
        print(f"referência {sample.reference_text}")
    if verdade:
        print(f"{VERDE}falado{RESET}     {VERDE}{verdade}{RESET}   (anotação humana)")
    if sample.sintetico:
        print(
            f"{CINZA}voz sintética ({sample.tts_voice}) — valida o pipeline, "
            f"não responde à pergunta sobre fala infantil{RESET}"
        )
    print("-" * 78)

    alvo = normalizar(verdade) if verdade else None
    referencia_norm = normalizar(sample.reference_text or "")

    async for r in run_engines(engines, req):
        if r.status is not EngineStatus.OK:
            print(
                f"  {r.name if hasattr(r, 'name') else r.engine:18} "
                f"{r.status.value} — {r.unavailable_reason}"
            )
            continue

        saida = normalizar(r.transcript or "")
        if alvo and saida == alvo:
            veredito = f"{VERDE}preservou a fala{RESET}"
        elif referencia_norm and saida == referencia_norm:
            veredito = f"{AMARELO}normalizou para a referência{RESET}"
        else:
            veredito = f"{CINZA}divergiu de ambas{RESET}"

        conf = r.metrics.get("confianca_media") or r.metrics.get("confianca_media_caractere")
        conf_txt = f"conf {conf:.3f}" if conf else ""
        print(f"  {r.engine:18} {r.latency_ms:>6}ms {conf_txt:>11}  {veredito}")
        print(f"  {'':18} → {r.transcript!r}")

    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparativo entre engines")
    parser.add_argument("--label", default="seed:metatese_problema")
    parser.add_argument("--todas", action="store_true", help="roda em todas as amostras")
    parser.add_argument("--engines", nargs="*", default=None)
    args = parser.parse_args()

    init_db()
    engine_names = args.engines or enabled_engine_names()

    with Session(get_engine()) as session:
        if args.todas:
            amostras = session.exec(select(Sample).order_by(Sample.id)).all()
        else:
            amostras = session.exec(select(Sample).where(Sample.label == args.label)).all()
            if not amostras:
                print(f"amostra '{args.label}' não encontrada; rode `make seed`", file=sys.stderr)
                return 1

        pares = []
        for s in amostras:
            anotacao = session.exec(select(Annotation).where(Annotation.sample_id == s.id)).first()
            pares.append((s, anotacao.transcript_fiel if anotacao else None))

    for sample, verdade in pares:
        asyncio.run(comparar(sample, verdade, engine_names))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
