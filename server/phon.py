"""
Camada FONÉTICA: reconhece SONS (fones IPA), não letras.

Modelo: facebook/wav2vec2-xlsr-53-espeak-cv-ft — reconhecedor de fonemas
multilíngue treinado com rótulos do espeak-ng. Como ele não tem léxico, não
"conserta" palavras: se a pessoa disse /pɾobɾema/, ele não escreve /pɾoblema/
só porque "problema" é uma palavra comum.

A frase esperada é fonetizada com o espeak-ng (pt-br) usando o MESMO
inventário de fones do modelo, e as duas sequências são alinhadas com custos
por classe de som (vogal↔vogal é leve; consoante trocada/omitida é erro).
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field, asdict

import numpy as np

PHON_MODEL_ID = os.environ.get("PHON_MODEL", "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
TARGET_SR = 16000

# phonemizer precisa achar a lib do espeak-ng (homebrew no mac)
for _cand in ("/opt/homebrew/lib/libespeak-ng.dylib", "/usr/local/lib/libespeak-ng.dylib",
              "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1", "/usr/lib/libespeak-ng.so.1"):
    if os.path.exists(_cand):
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", _cand)
        break


# ------------------------------------------------------------ classes de som
VOWEL_CHARS = set("aeiouyæɐɑɛəɪɨɔøœʊʌɤɯ")
R_GROUP = {"r", "ɾ", "x", "h", "ʁ", "χ", "ɹ", "ɣ", "ʀ", "ɽ"}
NASAL_C = {"n", "ɲ", "ŋ", "m", "ɱ"}
SIBIL_S = {"s", "ʃ"}
SIBIL_Z = {"z", "ʒ"}
VOICING = {("p", "b"), ("t", "d"), ("k", "ɡ"), ("f", "v"), ("s", "z"), ("ʃ", "ʒ"), ("tʃ", "dʒ")}
ALLOPHONE = {("ɡ", "ɣ"), ("b", "β"), ("d", "ð"), ("t", "tʃ"), ("d", "dʒ"), ("l", "ɫ"),
             ("j", "i"), ("j", "ɪ"), ("w", "u"), ("w", "ʊ"), ("l", "w"), ("l", "ʊ"), ("l", "u"),
             ("s", "ʃ"), ("z", "ʒ"), ("ɲ", "n"), ("k", "kʰ"), ("p", "pʰ"), ("t", "tʰ")}
_TILDE = "̃"


def base(ph: str) -> str:
    """remove marcas de duração/tom, mantém nasalização."""
    ph = ph.replace("ː", "").replace("ˈ", "").replace("ˌ", "")
    ph = re.sub(r"[0-9]", "", ph)
    return ph


def is_vowel(ph: str) -> bool:
    b = base(ph).replace(_TILDE, "")
    return bool(b) and b[0] in VOWEL_CHARS


def split_diphthongs(ph: str) -> list[str]:
    """'aʊ' -> ['a','ʊ']; 'ɐ̃ʊ̃' -> ['ɐ̃','ʊ̃']; consoantes compostas (tʃ, dʒ) ficam."""
    ph = base(ph)
    if not is_vowel(ph):
        return [ph]
    out, cur = [], ""
    for ch in ph:
        if ch == _TILDE:
            cur += ch
            continue
        if cur:
            out.append(cur)
        cur = ch
    if cur:
        out.append(cur)
    return out


def _pair_in(a: str, b: str, pairs: set) -> bool:
    return (a, b) in pairs or (b, a) in pairs


VOWEL_GROUPS = [set("aæɐɑ"), set("eɛəɪiyɨ"), set("oɔʊu")]
SIBILANT_SOFT = {("s", "z"), ("ʃ", "ʒ"), ("ʃ", "tʃ"), ("ʒ", "dʒ")}  # sandhi/dialeto/confusão do modelo
GLIDES = {"w", "j"}


def _vgroup(v: str) -> int:
    v0 = v.replace(_TILDE, "")
    for k, grp in enumerate(VOWEL_GROUPS):
        if v0 and v0[0] in grp:
            return k
    return -1


L_VOCALIZED = {"u", "ʊ", "w", "o"}


def sub_cost(e: str, g: str, nxt: str | None = None) -> float:
    """Custo 0..1 de ouvir g no lugar de e. < 0.5 = leve; >= 0.5 = erro.

    `nxt` é o próximo fone ESPERADO — precisa dele porque a mesma troca pode ser
    normal ou erro conforme a posição: /l/ virar /u/ é a vocalização comum do L
    que fecha sílaba ("azul" ~ /azuw/), mas antes de vogal ("chocolate") é desvio.
    """
    if e == g:
        return 0.0
    if e == "l" and g in L_VOCALIZED:
        return 0.2 if (nxt is None or not is_vowel(nxt)) else 1.0
    ev, gv = is_vowel(e), is_vowel(g)
    if ev and gv:
        # espeak escreve æ/y/ʊ onde o modelo ouve a/i/u: mesma família = quase igual
        if _vgroup(e) == _vgroup(g) and _vgroup(e) >= 0:
            return 0.05 if e.replace(_TILDE, "") != g.replace(_TILDE, "") else 0.1
        return 0.3                                # vogal de outra família: leve
    if ev != gv:
        # vogal ↔ semivogal/l final é comum (azul ~ azuw), resto é erro
        if _pair_in(e, g, ALLOPHONE):
            return 0.2
        if (e in GLIDES) or (g in GLIDES):
            return 0.25
        return 1.0
    # consoante ↔ consoante
    if e in R_GROUP and g in R_GROUP:
        return 0.15
    if _pair_in(e, g, ALLOPHONE):
        return 0.15
    if e in NASAL_C and g in NASAL_C:
        return 0.2
    if _pair_in(e, g, SIBILANT_SOFT):
        return 0.4
    if _pair_in(e, g, VOICING):
        return 1.0
    return 1.0


def del_cost(e: str) -> float:  # esperado e não ouvido
    if is_vowel(e):
        return 0.3 if e in ("ə", "ɨ", "y", "ʊ", "ɪ") else 0.45  # redução vocálica é comum
    if e in NASAL_C:
        return 0.3   # nasal em coda vira nasalização da vogal
    if e in GLIDES:
        return 0.35  # "pulou" ~ /pulo/
    return 1.0


def ins_cost(g: str) -> float:  # ouvido a mais
    if is_vowel(g) or g in GLIDES:
        return 0.3
    if g in NASAL_C or g in ("ə", "h"):
        return 0.35   # transições/aspirações que o modelo às vezes inventa
    return 0.7        # consoante a mais (inclui /r/: pode ser rotacismo)


# --------------------------------------------------------------- estruturas
@dataclass
class PhoneOp:
    op: str            # equal | sub | del | ins
    exp: str | None
    got: str | None
    cost: float
    word: int          # índice da palavra esperada


@dataclass
class WordPhon:
    index: int
    word: str
    expected: list[str]
    heard: list[str]
    ops: list[dict] = field(default_factory=list)
    cost: float = 0.0
    severity: str = "ok"     # ok | leve | erro
    credit: float = 1.0
    labels: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ modelo
class PhoneASR:
    def __init__(self, model_id: str = PHON_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.loaded = False
        self._lock = threading.Lock()
        self._phon_cache: dict[str, list[list[str]]] = {}

    def load(self) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        t0 = time.time()
        if self.device is None:
            env = os.environ.get("ASR_DEVICE", "").strip().lower()
            self.device = env or ("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[phon] carregando {self.model_id} em {self.device} ...", flush=True)
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_id).eval()
        try:
            model.to(self.device)
        except Exception as e:  # pragma: no cover
            print(f"[phon] falha em {self.device} ({e}); usando cpu", flush=True)
            self.device = "cpu"
            model.to("cpu")
        self.model = model
        # aquece o backend do espeak
        self.phonemize_words(["olá"])
        self.loaded = True
        print(f"[phon] pronto em {time.time() - t0:.1f}s", flush=True)

    # ---- fonetização da frase esperada (mesmo inventário do modelo)
    def _phonemize_one(self, text: str) -> list[list[str]]:
        """fonetiza um texto; devolve lista de palavras, cada uma lista de fones."""
        s = self.processor.tokenizer.phonemize(text, phonemizer_lang="pt-br")
        groups = [g.split() for g in s.split("|")]
        groups = [g for g in groups if g]
        out = []
        for g in groups:
            flat: list[str] = []
            for p in g:
                flat.extend(split_diphthongs(p))
            out.append(flat)
        return out

    def phonemize_words(self, words: list[str]) -> list[list[str]]:
        """Fonetiza a frase inteira (contexto importa: "é" sozinho vira "e agudo").
        Se o espeak não devolver o mesmo número de palavras, cai para palavra a palavra."""
        if not words:
            return []
        key = " ".join(words)
        if key in self._phon_cache:
            return self._phon_cache[key]
        groups = self._phonemize_one(key)
        if len(groups) != len(words):
            groups = []
            for w in words:
                g = self._phonemize_one(w)
                groups.append([p for grp in g for p in grp] if g else [])
        self._phon_cache[key] = groups
        return groups

    # ---- reconhecimento de fones
    def recognize(self, audio: np.ndarray, sr: int) -> list[str]:
        import torch
        from .asr import to_mono_float32, resample

        audio = to_mono_float32(audio)
        if sr != TARGET_SR:
            audio = resample(audio, sr, TARGET_SR)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / max(peak, 0.1) * 0.9
        inputs = self.processor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
        with self._lock, torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
        ids = logits.argmax(-1).cpu().unsqueeze(0)
        text = self.processor.batch_decode(ids)[0]
        phones: list[str] = []
        for p in text.split():
            phones.extend(split_diphthongs(p))
        return phones


# ------------------------------------------------------------- alinhamento
def align_phones(exp: list[tuple[str, int]], got: list[str]) -> list[PhoneOp]:
    """exp: [(fone, idx_palavra)], got: [fone]. DP com custos por classe."""
    n, m = len(exp), len(got)
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + del_cost(exp[i - 1][0]); B[i][0] = "del"
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + ins_cost(got[j - 1]); B[0][j] = "ins"
    for i in range(1, n + 1):
        e = exp[i - 1][0]
        nxt = exp[i][0] if i < n else None
        for j in range(1, m + 1):
            g = got[j - 1]
            c_sub = D[i - 1][j - 1] + sub_cost(e, g, nxt)
            c_del = D[i - 1][j] + del_cost(e)
            c_ins = D[i][j - 1] + ins_cost(g)
            best = min(c_sub, c_del, c_ins)
            D[i][j] = best
            B[i][j] = "sub" if best == c_sub else ("del" if best == c_del else "ins")

    ops: list[PhoneOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        b = B[i][j]
        if b == "sub":
            e, w = exp[i - 1]; g = got[j - 1]
            c = sub_cost(e, g, exp[i][0] if i < n else None)
            ops.append(PhoneOp("equal" if c == 0 else "sub", e, g, c, w))
            i, j = i - 1, j - 1
        elif b == "del":
            e, w = exp[i - 1]
            ops.append(PhoneOp("del", e, None, del_cost(e), w))
            i -= 1
        else:
            g = got[j - 1]
            # inserção: atribui à palavra do fone esperado anterior (ou próximo)
            w = exp[i - 1][1] if i > 0 else (exp[0][1] if exp else 0)
            ops.append(PhoneOp("ins", None, g, ins_cost(g), w))
            j -= 1
    ops.reverse()

    # Vogal inserida ao lado de uma vogal igual = alongamento ("a" longo vira /ɐɐ/),
    # não som a mais. Mesma coisa para consoante geminada na emenda entre palavras.
    for k, op in enumerate(ops):
        if op.op != "ins":
            continue
        viz = []
        if k > 0:
            viz += [ops[k - 1].exp, ops[k - 1].got]
        if k + 1 < len(ops):
            viz += [ops[k + 1].exp, ops[k + 1].got]
        if any(v and base(v).replace(_TILDE, "") == base(op.got).replace(_TILDE, "") for v in viz):
            op.cost = 0.05
    return ops


def _label(op: PhoneOp) -> str | None:
    e, g = op.exp, op.got
    if op.op == "equal" or op.cost < 0.5:
        return None
    if op.op == "sub":
        if e == "l" and g in R_GROUP:
            return f"Troca de L por R (rotacismo): som /l/ virou /{g}/"
        if e in R_GROUP and g == "l":
            return f"Troca de R por L (lambdacismo): som /{e}/ virou /l/"
        if _pair_in(e, g, VOICING):
            return f"Troca surda/sonora: som /{e}/ virou /{g}/"
        return f"Som /{e}/ virou /{g}/"
    if op.op == "del":
        return f"Som /{e}/ não foi ouvido (omissão)"
    if op.op == "ins":
        return f"Som /{g}/ a mais"
    return None


def compare_phones(words: list[str], expected: list[list[str]], heard: list[str]) -> dict:
    flat = [(p, wi) for wi, ph in enumerate(expected) for p in ph]
    ops = align_phones(flat, heard)

    per_word: list[WordPhon] = [WordPhon(i, w, expected[i], []) for i, w in enumerate(words)]
    for op in ops:
        wp = per_word[op.word] if 0 <= op.word < len(per_word) else None
        if wp is None:
            continue
        if op.got is not None:
            wp.heard.append(op.got)
        wp.ops.append(asdict(op))
        wp.cost += op.cost
        lab = _label(op)
        if lab and lab not in wp.labels:
            wp.labels.append(lab)

    total_cost = 0.0
    total_phones = 0
    credits = []
    for wp in per_word:
        n = max(len(wp.expected), 1)
        total_cost += wp.cost
        total_phones += n
        severe = any(o["cost"] >= 0.5 for o in wp.ops)
        rel = wp.cost / n
        if severe:
            wp.severity = "erro"
            wp.credit = round(min(0.6, max(0.0, 1.0 - rel)), 2)
        elif rel >= 0.2:
            wp.severity = "leve"
            wp.credit = 0.9
            if not wp.labels:   # nunca mostrar aviso sem explicação
                wp.labels.append("Pequena variação de vogal/sotaque (aceitável)")
        else:
            wp.severity = "ok"
            wp.credit = 1.0
        credits.append(wp.credit)

    nota = round(100.0 * (sum(credits) / len(credits)), 1) if credits else 0.0
    return {
        "esperado_ipa": [" ".join(w.expected) for w in per_word],
        "ouvido_ipa": " ".join(heard),
        "palavras": [asdict(w) for w in per_word],
        "nota_fonetica": nota,
        "fones_esperados": total_phones,
        "custo_total": round(total_cost, 2),
    }
