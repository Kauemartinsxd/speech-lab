from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app.audio.normalize import AudioNormalizationError, normalize_audio
from app.config import get_settings
from app.db import get_session
from app.models import Sample, SampleSource

router = APIRouter(prefix="/samples", tags=["samples"])


@router.get("")
def list_samples(session: Session = Depends(get_session)) -> dict:
    rows = session.exec(select(Sample).order_by(Sample.created_at.desc())).all()
    return {"samples": [r.model_dump() for r in rows]}


@router.post("")
async def create_sample(
    file: UploadFile = File(...),
    label: str = Form(""),
    reference_text: str | None = Form(None),
    source: str = Form(SampleSource.UPLOAD.value),
    session: Session = Depends(get_session),
) -> dict:
    """Recebe áudio de qualquer formato e normaliza para WAV mono 16 kHz PCM."""
    settings = get_settings()
    settings.abs_audio_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "audio").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        # nome provisório; o definitivo usa o sha256, para deduplicar
        staged = settings.abs_audio_dir / f"staging-{tmp_path.stem}.wav"
        normalized = normalize_audio(tmp_path, staged)

        final_path = settings.abs_audio_dir / f"{normalized.sha256[:16]}.wav"
        if final_path.exists():
            staged.unlink(missing_ok=True)
        else:
            staged.rename(final_path)
    except AudioNormalizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    sample = Sample(
        label=label or (file.filename or "sem rótulo"),
        source=SampleSource(source),
        original_filename=file.filename,
        audio_path=str(final_path),
        sha256=normalized.sha256,
        duration_s=normalized.duration_s,
        sample_rate=normalized.sample_rate,
        reference_text=(reference_text or None),
    )
    session.add(sample)
    session.commit()
    session.refresh(sample)
    return sample.model_dump()


@router.get("/{sample_id}")
def get_sample(sample_id: int, session: Session = Depends(get_session)) -> dict:
    sample = session.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="amostra não encontrada")
    return sample.model_dump()


@router.get("/{sample_id}/audio")
def get_audio(sample_id: int, session: Session = Depends(get_session)) -> FileResponse:
    sample = session.get(Sample, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="amostra não encontrada")
    if sample.audio_discarded_at is not None or not sample.audio_path:
        raise HTTPException(
            status_code=410,
            detail="áudio descartado por política de privacidade; só restam métricas",
        )
    path = Path(sample.audio_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="arquivo de áudio ausente no disco")
    return FileResponse(path, media_type="audio/wav")
