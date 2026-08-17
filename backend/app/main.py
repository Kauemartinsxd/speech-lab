"""speech-lab — API local.

Sem login, sem multi-tenant, sem RBAC: é ferramenta de bancada, roda na máquina
de quem está experimentando.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_health, routes_run, routes_samples
from app.config import get_settings
from app.db import init_db
from app.engines.registry import unload_all


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    init_db()
    settings.abs_audio_dir.mkdir(parents=True, exist_ok=True)
    # Nenhum modelo é carregado aqui de propósito: em 16 GB, carga sob demanda.
    yield
    unload_all()


app = FastAPI(
    title="speech-lab",
    description="Bancada de comparação de pipelines de análise de fala em pt-BR",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_samples.router)
app.include_router(routes_run.router)
