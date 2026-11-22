"""Utilitas teks: cleaning, normalisasi slang (kamus alay), deteksi cakupan, pengecekan span."""
import re
from functools import lru_cache

from .config import RAW


def clean_text(s):
    s = str(s)
    s = re.sub(r"\\x[0-9a-fA-F]{2}", " ", s)       # byte emoji literal
    s = s.replace("\\n", " ").replace("\\t", " ")
    s = re.sub(r"^(\s*-\s*)", "", s)
    s = re.sub(r"^RT\s+", "", s)
    s = re.sub(r"\b(USER)(\s+USER)+\b", "USER", s)   # rapatkan mention berulang
    s = re.sub(r"'$", "", s.strip())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_key(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@lru_cache(1)
def kamus():
    d = {}
    for line in open(RAW / "new_kamusalay.csv", encoding="latin-1").read().splitlines():
        if "," in line:
            a, b = line.split(",", 1)
            d[a.strip().lower()] = b.strip().lower()
    return d


def normalize_slang(text):
    k = kamus()
    toks = re.findall(r"\w+|[^\w\s]", str(text).lower())
    return " ".join(k.get(t, t) for t in toks)


def slang_ratio(text):
    toks = re.findall(r"[a-z]+", str(text).lower())
    if not toks:
        return 0.0
    k = kamus()
    return sum(t in k for t in toks) / len(toks)


_ID_WORDS = set("""yang dan di ke dari ini itu tidak gak ga nggak enggak aja saja sih deh dong kok lah kan
juga ada sudah udah udh lagi lg mau bisa jadi kalau kalo karena krn sama buat untuk utk orang ya yg dgn dengan
aku gue gua gw lu lo elo loe kamu kau dia kita kami mereka apa kenapa siapa mana bgt banget kayak kaya tapi
tp pak bu kak bang mas mbak jangan jgn bukan nya dasar semua pada udah belum blm masih cuma emang memang""".split())


def scope_check(text):
    """Heuristik cakupan: Indonesia/informal/campur bahasa = in-scope. Return (in_scope, reason)."""
    t = str(text).strip()
    if len(t) < 3:
        return False, "out_of_scope:empty_or_too_short"
    stripped = re.sub(r"\b(USER|URL)\b", "", t).strip()
    letters = re.findall(r"[A-Za-z]", stripped)
    if len(letters) < 3:
        return False, "out_of_scope:no_latin_text"
    if re.search(r"(\bdef \w+\(|\breturn \w|SELECT .* FROM|for \(|<div|^\{\s*\"|\+= )", t):
        return False, "out_of_scope:code"
    toks = re.findall(r"[a-z]+", stripped.lower())
    if not toks:
        return False, "out_of_scope:no_tokens"
    id_hits = sum(w in _ID_WORDS for w in toks)
    foreign_hits = sum(w in _FOREIGN_WORDS for w in toks)
    if id_hits == 0 and foreign_hits >= 3 and foreign_hits / len(toks) >= 0.3:
        return False, "out_of_scope:non_indonesian"
    return True, "in_scope"


_FOREIGN_WORDS = set("""the an is are were be been to of on at for with and or but not this that these those
you he she we they my your our their him her them its have has had does did will would can could should
please before after because while during about from into than then there here what which who when where how all
some any yes very really finally last next week yesterday today tomorrow enjoyed concert night band amazing
love like know think want need make take get go going come see look good great nice happy day days time people
friends family home work school book books park walk walks long little big new old first just only also well
because about would there their every never always something nothing everyone thank thanks sorry hello
los las del que una por mucho muy gusta caminar playa
les des est nous vous ils une pour avec sur aux soir voudrais
der das und ist ich wir ihr nicht ein eine mit auf heute keine habe vielleicht morgen zeit""".split())


def span_in_text(span, text):
    span = str(span).strip().lower()
    return len(span) >= 2 and span in str(text).lower()
