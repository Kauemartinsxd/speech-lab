"""
API + servidor estático da ferramenta de análise de fala.

  GET  /                -> interface
  GET  /api/status      -> modelos carregados?
  GET  /api/phrases     -> banco de frases
  POST /api/analyze     -> multipart: audio (arquivo), expected (texto)
"""
from __future__ import annotations

import base64
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .asr import LiteralASR, decode_audio_bytes
from .compare import compare, tokens
from .phon import PhoneASR, compare_phones
from .phrases import PHRASES

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI(title="Análise de Fala — Alfabetização")

orto = LiteralASR()
fon = PhoneASR()
_load_errors: dict[str, str] = {}

# Senha (HTTP Basic). Sem TTS_PASSWORD definida não há autenticação — é o modo
# local, em que o servidor só escuta em 127.0.0.1. Ao expor pela internet
# (cloudflared) defina TTS_PASSWORD: são gravações de voz, não podem ficar
# abertas para quem descobrir a URL.
_USER = os.environ.get("TTS_USER", "prof").strip()
_PASSWORD = os.environ.get("TTS_PASSWORD", "").strip()


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if _PASSWORD:
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            try:
                user, _, pwd = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                ok = secrets.compare_digest(user, _USER) and secrets.compare_digest(pwd, _PASSWORD)
            except Exception:  # noqa: BLE001
                ok = False
        if not ok:
            return Response(status_code=401, content="Acesso restrito.",
                            headers={"WWW-Authenticate": 'Basic realm="Leitura em voz alta"'})
    return await call_next(request)


def _load_all() -> None:
    for name, m in (("orto", orto), ("fon", fon)):
        try:
            m.load()
        except Exception as e:  # noqa: BLE001
            _load_errors[name] = f"{type(e).__name__}: {e}"
            print(f"[{name}] ERRO ao carregar: {e}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_all, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {
        "orto": {"loaded": orto.loaded, "model": orto.model_id, "device": orto.device, "error": _load_errors.get("orto")},
        "fon": {"loaded": fon.loaded, "model": fon.model_id, "device": fon.device, "error": _load_errors.get("fon")},
        "ready": orto.loaded or fon.loaded,
    }


@app.get("/api/phrases")
def phrases():
    return PHRASES


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...), expected: str = Form(...)):
    if not (orto.loaded or fon.loaded):
        raise HTTPException(503, "Modelos ainda carregando — aguarde alguns segundos.")
    expected = (expected or "").strip()
    if not expected:
        raise HTTPException(400, "Informe a frase esperada.")
    data = await audio.read()
    if len(data) < 1000:
        raise HTTPException(400, "Áudio vazio ou muito curto.")
    try:
        wave, sr = decode_audio_bytes(data, audio.filename or "audio")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Não consegui decodificar o áudio: {e}")
    if len(wave) < sr * 0.3:
        raise HTTPException(400, "Gravação muito curta (menos de 0,3 s).")

    t0 = time.time()
    result = build_result(expected, wave, sr)
    result["tempo_total_s"] = round(time.time() - t0, 2)
    return JSONResponse(result)


# ---------------------------------------------------------------- combinação
def build_result(expected: str, wave, sr: int) -> dict:
    words = tokens(expected)
    n = max(len(words), 1)

    # --- camada 1: letras (transcrição literal)
    orto_res = None
    tr = None
    if orto.loaded:
        tr = orto.transcribe(wave, sr)
        orto_res = compare(expected, tr.text)

    # --- camada 2: sons (fonemas)
    fon_res = None
    heard_ph: list[str] = []
    if fon.loaded:
        heard_ph = fon.recognize(wave, sr)
        exp_ph = fon.phonemize_words(words)
        fon_res = compare_phones(words, exp_ph, heard_ph)

    # --- por palavra esperada
    per_word = []
    for i, w in enumerate(words):
        per_word.append({"index": i, "palavra": w, "letras": None, "sons": None,
                         "status": "ok", "credit": 1.0, "rotulos": []})
    extras = []
    if orto_res:
        wi = 0
        for op in orto_res["palavras"]:
            if op["op"] == "ins":
                extras.append({"ouvido": op["heard"], "rotulos": op["labels"]})
                continue
            entry = per_word[wi]
            entry["letras"] = {
                "op": op["op"], "ouvido": op["heard"], "diff": op["char_diff"],
                "rotulos": op["labels"], "severidade": op["severity"], "credit": op["credit"],
            }
            wi += 1
    if fon_res:
        for i, wp in enumerate(fon_res["palavras"]):
            per_word[i]["sons"] = {
                "esperado_ipa": " ".join(wp["expected"]), "ouvido_ipa": " ".join(wp["heard"]),
                "ops": wp["ops"], "rotulos": wp["labels"], "severidade": wp["severity"], "credit": wp["credit"],
            }

    sev_rank = {"ok": 0, "leve": 1, "suspeita": 2, "erro": 3}
    n_ok = 0
    credit_sum = 0.0
    erros = []
    for e in per_word:
        credits, labels = [], []
        sev_l = e["letras"]["severidade"] if e["letras"] else None
        sev_s = e["sons"]["severidade"] if e["sons"] else None
        if e["letras"]:
            credits.append(e["letras"]["credit"]); labels += e["letras"]["rotulos"]
        if e["sons"]:
            credits.append(e["sons"]["credit"]); labels += e["sons"]["rotulos"]

        # Combinação assimétrica. A camada de SONS não tem dicionário: foi feita
        # para ser fiel, então ela decide sobre pronúncia. A de LETRAS tem viés
        # lexical (pode "consertar") e erra a transcrição com frequência, então
        # sozinha ela levanta SUSPEITA — não condena. Quando as duas apontam o
        # mesmo desvio, é erro confirmado.
        if sev_s == "erro":
            e["status"] = "erro"
            e["credit"] = round(min(credits), 2)
        elif sev_l == "erro" and sev_s in ("ok", "leve"):
            e["status"] = "suspeita"
            e["credit"] = min(0.85, e["sons"]["credit"])
            labels.append("Só a camada de letras ouviu diferente — pode ser falha do reconhecedor; ouça a gravação")
        elif sev_l == "erro" and sev_s is None:      # sem camada de sons
            e["status"] = "erro"
            e["credit"] = round(min(credits), 2)
        elif "leve" in (sev_l, sev_s):
            e["status"] = "leve"
            e["credit"] = round(min(credits), 2)
        else:
            e["status"] = "ok"
            e["credit"] = round(min(credits), 2) if credits else 1.0
        # dedup mantendo ordem
        seen = set(); e["rotulos"] = [l for l in labels if not (l in seen or seen.add(l))]
        credit_sum += e["credit"]
        if e["status"] == "ok":
            n_ok += 1
        else:
            erros.append({
                "palavra": e["palavra"], "status": e["status"],
                "ouvido_letras": e["letras"]["ouvido"] if e["letras"] else None,
                "diff": e["letras"]["diff"] if e["letras"] else [],
                "esperado_ipa": e["sons"]["esperado_ipa"] if e["sons"] else None,
                "ouvido_ipa": e["sons"]["ouvido_ipa"] if e["sons"] else None,
                "rotulos": e["rotulos"],
            })
    for x in extras:
        # palavra extra só existe na camada de letras -> suspeita, não erro
        erros.append({"palavra": None, "status": "suspeita", "ouvido_letras": x["ouvido"], "diff": [],
                      "esperado_ipa": None, "ouvido_ipa": None,
                      "rotulos": x["rotulos"] + ["Pode ser hesitação, ruído ou falha do reconhecedor"]})

    nota = 100.0 * credit_sum / n - 100.0 * 0.25 * len(extras) / n
    nota = round(max(0.0, min(100.0, nota)), 1)

    heard_text = tr.text if tr else ""
    if not heard_text and not heard_ph:
        veredito = "Nada foi reconhecido — grave de novo, mais perto do microfone."
    elif nota >= 95:
        veredito = "Excelente! Fala clara e fiel à frase."
    elif nota >= 85:
        veredito = "Muito bom — pequenos desvios."
    elif nota >= 65:
        veredito = "Razoável — há trocas ou omissões que merecem atenção."
    else:
        veredito = "Vários desvios em relação à frase — vale repetir com calma."

    return {
        "esperado": expected,
        "palavras": per_word,
        "extras": extras,
        "erros": erros,
        "letras": {
            "ouvido": heard_text,
            "nota": orto_res["metricas"]["nota"] if orto_res else None,
            "metricas": orto_res["metricas"] if orto_res else None,
            "modelo": orto.model_id, "confianca": [c for c in tr.chars] if tr else [],
        } if orto.loaded else None,
        "sons": {
            "ouvido_ipa": " ".join(heard_ph),
            "esperado_ipa": fon_res["esperado_ipa"] if fon_res else [],
            "nota": fon_res["nota_fonetica"] if fon_res else None,
            "modelo": fon.model_id,
        } if fon.loaded else None,
        "metricas": {
            "nota": nota,
            "palavras_ok": n_ok,
            "palavras_total": len(words),
            "palavras_extras": len(extras),
            "nota_letras": orto_res["metricas"]["nota"] if orto_res else None,
            "nota_sons": fon_res["nota_fonetica"] if fon_res else None,
            "wer": orto_res["metricas"]["wer"] if orto_res else None,
            "cer": orto_res["metricas"]["cer"] if orto_res else None,
        },
        "veredito": veredito,
        "audio": {"duracao_s": round(len(wave) / sr, 2), "sr": sr},
    }


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
