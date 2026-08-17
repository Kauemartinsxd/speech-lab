from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.db import get_session, get_engine as get_db_engine
from app.engines.base import AudioInput, EngineRequest, EngineResult
from app.engines.registry import enabled_engine_names, load_engine
from app.engines.runner import run_engines
from app.models import EngineOutcome, Run, RunStatus, Sample

log = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"])


class CreateRunRequest(BaseModel):
    sample_id: int
    engines: list[str] | None = None
    """None => todas as habilitadas no .env."""
    reference_text: str | None = None
    options: dict = {}


def _serialize(result: EngineResult) -> dict:
    return dataclasses.asdict(result)


@router.post("")
def create_run(body: CreateRunRequest, session: Session = Depends(get_session)) -> dict:
    sample = session.get(Sample, body.sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="amostra não encontrada")

    requested = body.engines if body.engines is not None else enabled_engine_names()
    run = Run(
        sample_id=sample.id,
        engines_requested=requested,
        options=body.options or {},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    if body.reference_text is not None and body.reference_text != sample.reference_text:
        sample.reference_text = body.reference_text or None
        session.add(sample)
        session.commit()

    return {"run_id": run.id, "engines": requested}


@router.get("/{run_id}/stream")
async def stream_run(run_id: int) -> EventSourceResponse:
    """Executa as engines e entrega cada resultado assim que fica pronto.

    Não espera a mais lenta: é isso que faz o comparativo aparecer na tela em
    ondas, com o CTC chegando antes do Whisper large-v3.
    """
    with Session(get_db_engine()) as session:
        run = session.get(Run, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="execução não encontrada")
        sample = session.get(Sample, run.sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="amostra não encontrada")
        if not sample.audio_path or sample.audio_discarded_at is not None:
            raise HTTPException(
                status_code=410,
                detail="áudio descartado por política de privacidade; re-execução impossível",
            )

        engine_names = list(run.engines_requested)
        options = dict(run.options or {})
        audio = AudioInput(
            path=Path(sample.audio_path),
            sample_rate=sample.sample_rate,
            duration_s=sample.duration_s,
            sha256=sample.sha256,
        )
        reference_text = sample.reference_text
        sample_id = sample.id

    settings = get_settings()
    options.setdefault("gop_threshold", settings.gop_threshold)

    engines = [e for e in (load_engine(n) for n in engine_names) if e is not None]
    req = EngineRequest(audio=audio, reference_text=reference_text, options=options)

    async def event_source():
        yield {
            "event": "start",
            "data": json.dumps(
                {"run_id": run_id, "engines": [e.name for e in engines]},
                ensure_ascii=False,
            ),
        }

        try:
            async for result in run_engines(engines, req):
                payload = _serialize(result)
                _persist(run_id, sample_id, result, payload)
                yield {
                    "event": "result",
                    "data": json.dumps(payload, ensure_ascii=False, default=str),
                }
        except Exception as exc:
            log.exception("stream da execução %s falhou", run_id)
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(exc)}, ensure_ascii=False),
            }
            _finish(run_id, RunStatus.FAILED)
            return

        _finish(run_id, RunStatus.DONE)
        yield {"event": "done", "data": json.dumps({"run_id": run_id})}

    return EventSourceResponse(event_source())


def _persist(run_id: int, sample_id: int, result: EngineResult, payload: dict) -> None:
    with Session(get_db_engine()) as session:
        session.add(
            EngineOutcome(
                run_id=run_id,
                sample_id=sample_id,
                engine=result.engine,
                status=result.status.value,
                unavailable_reason=result.unavailable_reason,
                transcript=result.transcript,
                latency_ms=result.latency_ms,
                cost_estimate_usd=result.cost_estimate_usd,
                model_version=result.model_version,
                device=result.device,
                payload=json.loads(json.dumps(payload, default=str)),
            )
        )
        session.commit()


def _finish(run_id: int, status: RunStatus) -> None:
    with Session(get_db_engine()) as session:
        run = session.get(Run, run_id)
        if run is not None:
            run.status = status
            run.finished_at = datetime.utcnow()
            session.add(run)
            session.commit()
