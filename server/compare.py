"""
Comparação entre a frase esperada e a transcrição literal.

- alinhamento palavra a palavra (Levenshtein ponderado pela semelhança de letras)
- diff letra a letra em cada palavra trocada
- classificação do tipo de erro (troca L/R, omissão de S final, surda/sonora...)
- pontuação: palavras corretas, WER, similaridade de caracteres e nota final
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field, asdict

# ------------------------------------------------------------- normalização
_PUNCT_RE = re.compile(r"[^\w\s'-]", flags=re.UNICODE)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = _PUNCT_RE.sub(" ", text)
    text = text.replace("-", " ")
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


# --------------------------------------------------------- distância de edição
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_dist_norm(a: str, b: str) -> float:
    m = max(len(a), len(b), 1)
    return levenshtein(a, b) / m


# ---------------------------------------------------------- alinhamento words
@dataclass
class WordOp:
    op: str  # equal | sub | del (omitida) | ins (extra)
    expected: str | None
    heard: str | None
    char_diff: list[dict] = field(default_factory=list)  # [{"tag": "equal|replace|delete|insert", "exp": "l", "got": "r"}]
    labels: list[str] = field(default_factory=list)
    severity: str = "ok"  # ok | leve | erro
    credit: float = 1.0


def best_split(exp_a: str, exp_b: str, got: str) -> tuple[str, str]:
    """Divide `got` no ponto que melhor casa com exp_a + exp_b ("osmenino" -> "os","menino")."""
    best_d, best_k = None, 0
    for k in range(len(got) + 1):
        d = levenshtein(strip_accents(exp_a), strip_accents(got[:k])) + \
            levenshtein(strip_accents(exp_b), strip_accents(got[k:]))
        if best_d is None or d < best_d:
            best_d, best_k = d, k
    return got[:best_k], got[best_k:]


def align_words(exp: list[str], got: list[str]) -> list[WordOp]:
    """Alinha palavras esperadas × ouvidas.

    Além de trocar/omitir/inserir, aceita JUNTAR e SEPARAR palavras: o CTC decide
    sozinho onde pôr o espaço e erra muito ("os meninos" -> "osmenino",
    "cachorro" -> "ca chorro"), o que não é erro de fala e não pode virar erro aqui.
    """
    n, m = len(exp), len(got)
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        D[i][0] = i
        B[i][0] = "del"
    for j in range(1, m + 1):
        D[0][j] = j
        B[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e, g = exp[i - 1], got[j - 1]
            if e == g:
                sub = D[i - 1][j - 1]
                sub_op = "equal"
            else:
                # custo de substituição entre 0.35 e 1.0 conforme semelhança;
                # palavras muito diferentes custam 1.5 (ainda < del+ins = 2, mas
                # perdem para "omitida + extra" quando há uma palavra igual perto)
                d = char_dist_norm(strip_accents(e), strip_accents(g))
                sub = D[i - 1][j - 1] + (1.5 if d >= 0.6 else 0.35 + 0.65 * d)
                sub_op = "sub"
            dele = D[i - 1][j] + 1.0
            ins = D[i][j - 1] + 1.0
            cand = [(sub, sub_op), (dele, "del"), (ins, "ins")]
            # Juntar/separar usa distância ABSOLUTA (não normalizada): assim uma
            # palavrinha ("o") não é engolida de graça só por ser curta — cada
            # letra que falta custa ~1, como uma palavra omitida.
            if i >= 2:   # duas palavras esperadas ouvidas grudadas numa só
                d2 = levenshtein(strip_accents(exp[i - 2] + exp[i - 1]), strip_accents(g))
                cand.append((D[i - 2][j - 1] + 0.15 + d2, "merge"))
            if j >= 2:   # uma palavra esperada partida em duas pelo ASR
                d2 = levenshtein(strip_accents(e), strip_accents(got[j - 2] + got[j - 1]))
                cand.append((D[i - 1][j - 2] + 0.15 + d2, "split"))
            best, best_op = min(cand, key=lambda c: c[0])
            D[i][j] = best
            B[i][j] = best_op

    ops: list[WordOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = B[i][j]
        if op == "merge":                       # got[j-1] cobre exp[i-2] e exp[i-1]
            ga, gb = best_split(exp[i - 2], exp[i - 1], got[j - 1])
            for e_w, g_w in ((exp[i - 1], gb), (exp[i - 2], ga)):
                if not g_w:      # nada sobrou para essa palavra: ela não foi dita
                    ops.append(WordOp("del", e_w, None))
                else:
                    ops.append(WordOp("equal" if e_w == g_w else "sub", e_w, g_w))
            i, j = i - 2, j - 1
        elif op == "split":                     # got[j-2]+got[j-1] cobrem exp[i-1]
            g_w = got[j - 2] + got[j - 1]
            ops.append(WordOp("equal" if exp[i - 1] == g_w else "sub", exp[i - 1], g_w))
            i, j = i - 1, j - 2
        elif op in ("equal", "sub"):
            ops.append(WordOp(op, exp[i - 1], got[j - 1]))
            i, j = i - 1, j - 1
        elif op == "del":
            ops.append(WordOp("del", exp[i - 1], None))
            i -= 1
        else:
            ops.append(WordOp("ins", None, got[j - 1]))
            j -= 1
    ops.reverse()
    return ops


# ------------------------------------------------------------- diff de letras
def char_diff(exp: str, got: str) -> list[dict]:
    sm = difflib.SequenceMatcher(None, exp, got, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        out.append({"tag": tag, "exp": exp[i1:i2], "got": got[j1:j2]})
    return out


# ------------------------------------------------- classificação de erros
_SURDA_SONORA = {("p", "b"), ("t", "d"), ("f", "v"), ("c", "g"), ("q", "g"), ("s", "z"), ("x", "j"), ("ch", "j"), ("x", "g"), ("ch", "g")}
_VOGAIS = set("aeiouáéíóúâêôãõà")
BOUNDARY_LABEL = "Vogal alongada ou hesitação na emenda entre as palavras (não é erro de pronúncia)"


def _pair(a: str, b: str) -> bool:
    return (a, b) in _SURDA_SONORA or (b, a) in _SURDA_SONORA


def classify(exp: str, got: str, diff: list[dict],
             prev_exp: str | None = None, next_exp: str | None = None) -> tuple[list[str], str]:
    """Retorna (rótulos, severidade) para uma troca de palavra."""
    labels: list[str] = []
    e_plain, g_plain = strip_accents(exp), strip_accents(got)
    n_diff = len(diff)
    boundary_only = True   # todos os desvios são artefato de emenda entre palavras?

    if e_plain == g_plain:
        return ["Só a acentuação/vogal aberta-fechada difere"], "leve"

    if sorted(e_plain) == sorted(g_plain):
        return ["Inversão de letras (metátese)"], "erro"

    for idx, d in enumerate(diff):
        if d["tag"] == "equal":
            continue
        e, g = strip_accents(d["exp"]), strip_accents(d["got"])
        tag = d["tag"]
        # Vogal a mais colada na emenda com a palavra vizinha: o ASR partiu uma
        # vogal longa ou houve hesitação ("a planta" -> "a aplanta"). Não é
        # epêntese (essa acontece DENTRO da palavra: pneu -> pineu).
        if tag == "insert" and (idx == 0 or idx == n_diff - 1):
            viz = strip_accents(prev_exp or "") if idx == 0 else strip_accents(next_exp or "")
            if all(c in _VOGAIS for c in g) and (g == viz or len(g) == 1):
                labels.append(BOUNDARY_LABEL)
                continue
        boundary_only = False
        if tag == "replace":
            if e == "l" and g == "r":
                labels.append("Troca de L por R (rotacismo)")
            elif e == "r" and g == "l":
                labels.append("Troca de R por L (lambdacismo)")
            elif e in ("rr", "r") and g in ("r", "rr"):
                labels.append("Troca de R forte/fraco")
            elif _pair(e, g):
                labels.append(f"Troca surda/sonora ({e} → {g})")
            elif e and g and e in _VOGAIS and g in _VOGAIS:
                labels.append(f"Troca de vogal ({e} → {g})")
            elif e in ("s", "ss", "ç", "c") and g in ("s", "ss", "ç", "c", "z"):
                labels.append("Troca de som de S")
            elif e in ("x", "ch") and g in ("x", "ch"):
                labels.append("Troca de X/CH")
            elif (e == "nh" and g == "n") or (e == "lh" and g == "l"):
                labels.append(f"Simplificação de {e.upper()} → {g.upper()}")
            else:
                labels.append(f"Substituição ({e} → {g})")
        elif tag == "delete":
            # letra(s) que estavam na palavra certa e não foram ditas
            pos_end = exp.endswith(d["exp"]) and d["exp"] != ""
            if e == "s" and pos_end:
                labels.append("Omissão do S final")
            elif e == "r" and pos_end:
                labels.append("Omissão do R final")
            elif e in ("m", "n") and pos_end:
                labels.append(f"Omissão da nasal final ({e})")
            elif e in ("r", "l") and not pos_end:
                labels.append(f"Omissão do {e.upper()} em encontro consonantal")
            elif e in _VOGAIS:
                labels.append(f"Omissão de vogal ({e})")
            elif len(e) >= 2:
                labels.append(f"Omissão de sílaba/trecho ({e})")
            else:
                labels.append(f"Omissão de letra ({e})")
        elif tag == "insert":
            if g in _VOGAIS:
                labels.append(f"Epêntese: acréscimo de vogal ({g})")
            elif len(g) >= 2:
                labels.append(f"Acréscimo de trecho ({g})")
            else:
                labels.append(f"Acréscimo de letra ({g})")

    dist = char_dist_norm(e_plain, g_plain)
    if boundary_only and labels:
        sev = "leve"
    elif dist >= 0.6:
        labels = [f"Palavra diferente ({exp} → {got})"]
        sev = "erro"
    else:
        sev = "erro"
    # deduplica mantendo ordem
    seen, uniq = set(), []
    for l in labels:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    return uniq or ["Substituição"], sev


# ------------------------------------------------------------------ resultado
def compare(expected: str, heard: str) -> dict:
    exp_t, got_t = tokens(expected), tokens(heard)
    ops = align_words(exp_t, got_t)

    n_exp = max(len(exp_t), 1)
    correct = 0
    subs = dels = ins = 0
    credit_total = 0.0
    errors: list[dict] = []

    for k, op in enumerate(ops):
        if op.op == "equal":
            correct += 1
            op.severity, op.credit = "ok", 1.0
            credit_total += 1.0
        elif op.op == "sub":
            subs += 1
            op.char_diff = char_diff(op.expected, op.heard)
            prev_e = next((o.expected for o in reversed(ops[:k]) if o.expected), None)
            next_e = next((o.expected for o in ops[k + 1:] if o.expected), None)
            op.labels, op.severity = classify(op.expected, op.heard, op.char_diff, prev_e, next_e)
            if BOUNDARY_LABEL in op.labels:
                op.credit = 1.0        # artefato do ASR, não desconta nota
            elif op.severity == "leve":
                op.credit = 0.85
            else:
                sim = 1.0 - char_dist_norm(strip_accents(op.expected), strip_accents(op.heard))
                op.credit = round(max(0.0, sim - 0.5), 2)  # crédito parcial pequeno
            credit_total += op.credit
            errors.append({
                "index": k, "tipo": "troca", "esperado": op.expected, "ouvido": op.heard,
                "rotulos": op.labels, "severidade": op.severity, "diff": op.char_diff,
            })
        elif op.op == "del":
            dels += 1
            op.severity, op.credit = "erro", 0.0
            op.labels = ["Palavra omitida (não foi dita)"]
            errors.append({"index": k, "tipo": "omissao", "esperado": op.expected, "ouvido": None,
                           "rotulos": op.labels, "severidade": "erro", "diff": []})
        else:  # ins
            ins += 1
            op.severity, op.credit = "erro", 0.0
            op.labels = ["Palavra extra (não estava na frase)"]
            errors.append({"index": k, "tipo": "insercao", "esperado": None, "ouvido": op.heard,
                           "rotulos": op.labels, "severidade": "erro", "diff": []})

    wer = (subs + dels + ins) / n_exp
    exp_join, got_join = strip_accents(" ".join(exp_t)), strip_accents(" ".join(got_t))
    cer = levenshtein(exp_join, got_join) / max(len(exp_join), 1)
    char_sim = max(0.0, 1.0 - cer)
    word_acc = correct / n_exp

    # nota: crédito por palavra (com parcial) menos penalidade por palavras extras
    nota = 100.0 * (credit_total / n_exp) - 100.0 * 0.5 * ins / n_exp
    nota = round(max(0.0, min(100.0, nota)), 1)

    if not got_t:
        veredito = "Nada foi reconhecido. Grave de novo, mais perto do microfone."
    elif nota >= 95:
        veredito = "Excelente! Fala clara e fiel à frase."
    elif nota >= 80:
        veredito = "Muito bom, com pequenos desvios."
    elif nota >= 60:
        veredito = "Razoável. Há trocas ou omissões que merecem atenção."
    else:
        veredito = "Vários desvios em relação à frase. Vale repetir com calma."

    return {
        "esperado": expected,
        "esperado_norm": " ".join(exp_t),
        "ouvido": heard,
        "ouvido_norm": " ".join(got_t),
        "palavras": [asdict(o) for o in ops],
        "erros": errors,
        "metricas": {
            "nota": nota,
            "palavras_corretas": correct,
            "palavras_total": len(exp_t),
            "acuracia_palavras": round(100 * word_acc, 1),
            "wer": round(100 * wer, 1),
            "cer": round(100 * cer, 1),
            "similaridade_caracteres": round(100 * char_sim, 1),
            "trocas": subs, "omissoes": dels, "insercoes": ins,
        },
        "veredito": veredito,
    }


if __name__ == "__main__":  # teste rápido
    import json
    print(json.dumps(compare("Qual é o problema dessa frase?", "qual é o probrema dessa frase"), ensure_ascii=False, indent=1))
