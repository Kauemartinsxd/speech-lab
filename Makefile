.PHONY: help setup check dev backend frontend test eval seed fmt clean

help:
	@echo "speech-lab"
	@echo ""
	@echo "  make setup    instala dependências (uv + npm)"
	@echo "  make check    verifica ffmpeg, espeak-ng, modelos, chaves"
	@echo "  make dev      backend :8000 + frontend :5173"
	@echo "  make test     pytest"
	@echo "  make seed     gera as amostras de exemplo"
	@echo "  make compare  comparativo Whisper vs CTC em todas as amostras"
	@echo "  make eval     WER/PER/F1 de cada engine contra a anotação humana"
	@echo "  make fmt      ruff format + check"

setup:
	uv sync --group dev
	cd frontend && npm install

check:
	uv run python scripts/check_env.py

dev:
	@echo "backend em http://localhost:8000  |  frontend em http://localhost:5173"
	@trap 'kill 0' INT TERM; \
	uv run uvicorn app.main:app --app-dir backend --reload --port 8000 & \
	(cd frontend && npm run dev) & \
	wait

backend:
	uv run uvicorn app.main:app --app-dir backend --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	uv run pytest -v

seed:
	uv run python scripts/seed.py

compare:
	uv run python scripts/compare.py --todas

eval:
	uv run python scripts/eval.py

fmt:
	uv run ruff format backend scripts
	uv run ruff check --fix backend scripts

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
