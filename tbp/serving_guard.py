"""Guard di lapisan serving: sanitasi input, filter prompt-injection, & redaksi PII sebelum diproses/dicatat."""
import re

MAX_CHARS = 2000

# Pola generik instruksi yang mencoba mengendalikan model (bukan isi tweet).
# v2 (ditambahkan setelah red-team v1; lihat LAPORAN bagian red-team).
INJECTION_PATTERNS = [
    r"(abaikan|lupakan|ignore|disregard|forget)\b[^.!?\n]{0,60}\b(instruksi|perintah|aturan|instruction|instructions|rules|prompt)[^.!?\n]*[.!?]?",
    r"\b(system|assistant|developer)\s*:[^.!?\n]*[.!?]?",
    r"\b(jawab|jawablah|set|setel|ubah)\b[^.!?\n]{0,30}\b(semua label|label|abstain)\b[^.!?\n]*[.!?]?",
    r"\{[^{}]*\"(HS|Abusive|HS_Individual|HS_Group|abstain)\"[^{}]*\}[^.!?\n]*[.!?]?",
]


def filter_injection(text):
    flags = []
    t = text
    for p in INJECTION_PATTERNS:
        if re.search(p, t, re.I):
            flags.append(p[:25])
            t = re.sub(p, " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip(), flags


def sanitize_input(text, injection_filter=True):
    t = str(text or "")
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", t)
    notes = []
    if len(t) > MAX_CHARS:
        notes.append(f"input dipotong dari {len(t)} ke {MAX_CHARS} karakter")
        t = t[:MAX_CHARS]
    if injection_filter:
        t2, flags = filter_injection(t)
        if flags:
            notes.append(f"injection filter: {len(flags)} pola dinetralkan")
            t = t2
    return t, "; ".join(notes)


def redact_pii(text):
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", str(text))
    t = re.sub(r"\b\d{16}\b", "[NIK]", t)
    t = re.sub(r"(\+62|62|0)8\d{7,11}\b", "[PHONE]", t)
    t = re.sub(r"@\w+", "USER", t)
    t = re.sub(r"https?://\S+", "URL", t)
    return t
