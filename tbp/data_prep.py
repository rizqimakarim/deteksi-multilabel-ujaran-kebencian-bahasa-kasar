"""Persiapan data: cleaning, dedup/near-dup, split terkunci (SHA-256), item out-of-scope,
corpus (contoh train + chunk guideline 3 strategi), source register, lexicon, dan qrels.

Jalankan:  python -m tbp.data_prep
"""
import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2

from .config import (DATA, GUIDE_DIR, LABELS, N_DEV_INSCOPE, N_TEST_INSCOPE, PROC, RAW, SEED)
from .textutil import clean_text, norm_key


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sha256_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Item di luar cakupan (unanswerable / wajib abstain). Ditulis sendiri oleh tim (bebas lisensi).
OOS_TEXTS = [
    # Inggris penuh
    "I really enjoyed the concert last night, the band was amazing.",
    "Please send me the quarterly report before Friday afternoon.",
    "The weather in London has been rainy for the whole week.",
    "Can you recommend a good book about machine learning?",
    "Our flight was delayed for three hours because of the storm.",
    "She finally finished her marathon training program yesterday.",
    "This recipe needs two cups of flour and one egg.",
    "The museum opens at nine and closes at five on weekdays.",
    "He is learning to play the guitar during the holidays.",
    "Remember to water the plants while I am away.",
    # Bahasa lain
    "今日はとても良い天気ですね。",
    "مرحبا كيف حالك اليوم",
    "Me gusta mucho caminar por la playa al atardecer.",
    "Je voudrais un café et un croissant, s'il vous plaît.",
    "Ich habe heute keine Zeit, vielleicht morgen.",
    "Ce soir nous allons au cinéma avec des amis.",
    # kode / url / angka / simbol / kosong
    "def add(a, b):\n    return a + b",
    "SELECT * FROM users WHERE id = 10;",
    "URL URL URL",
    "1234567890 0987654321",
    "!!!??? ... ### ***",
    "😂😂😂🔥🔥",
    "",
    "   ",
    "for (int i = 0; i < n; i++) { sum += i; }",
    "3.14159 2.71828 1.41421",
    "<div class='box'></div>",
    "==== ---- ====",
    "{\"key\": \"value\", \"n\": 3}",
    "USER USER USER",
]
# Daftar di atas TELAH dilihat saat menyetel textutil.scope_check -> dipakai sebagai set DEV (tuning).
# Daftar di bawah ditulis SETELAH scope_check dibekukan dan TIDAK pernah dipakai untuk menyetel apa pun
# -> satu-satunya teks out-of-scope di locked TEST (perbaikan leakage; lihat LAPORAN keterbatasan).
OOS_TEXTS_TEST = [
    "Could you tell me where the nearest train station is?",
    "My brother bought a new bicycle for his birthday last weekend.",
    "The meeting has been moved to Thursday at three o'clock.",
    "We watched a documentary about whales in the Pacific Ocean.",
    "Please remember to bring your umbrella because it might rain.",
    "Our team finally won the championship after ten long years.",
    "오늘 저녁에 같이 밥 먹을래요?",
    "Сегодня очень холодно на улице.",
    "Hôm nay trời đẹp quá, chúng ta đi dạo nhé.",
    "Magandang umaga sa inyong lahat, kumain na ba kayo?",
    "¿Dónde está la biblioteca más cercana?",
    "Domani andiamo al mare con tutta la famiglia.",
    "import numpy as np\nx = np.zeros(10)",
    "<html><body><h1>Hello</h1></body></html>",
    "SELECT name, age FROM students ORDER BY age;",
    "[1, 2, 3, 4, 5]",
    "+++ --- +++",
    "2024-01-15 08:30:00",
    "🙏🙏🙏",
    ".",
]


def load_main():
    df = pd.read_csv(RAW / "re_dataset.csv", encoding="latin-1")
    df = df[["Tweet"] + LABELS].rename(columns={"Tweet": "raw"})
    df["text"] = df["raw"].map(clean_text)
    df = df[df["text"].str.len() >= 3].copy()
    # dedup exact (pada teks ternormalisasi)
    df["nkey"] = df["text"].map(norm_key)
    n0 = len(df)
    # Teks sama dengan label berbeda = konflik anotasi -> dicatat dan dibuang
    conflict = df.groupby("nkey")[LABELS].nunique().max(axis=1)
    conflict_keys = set(conflict[conflict > 1].index)
    df = df[~df["nkey"].isin(conflict_keys)]
    df = df.drop_duplicates("nkey").reset_index(drop=True)
    stats = {"raw_rows": int(n0), "exact_dup_removed": int(n0 - len(df) - 0),
             "label_conflict_keys_removed": len(conflict_keys), "after_exact_dedup": int(len(df))}
    df["id"] = ["T%05d" % i for i in range(len(df))]
    df["n_labels"] = df[LABELS].sum(axis=1)
    return df, stats


def stratified_sample(df, n, seed):
    """Sampel stratified berdasarkan kombinasi label (label-powerset)."""
    rng = np.random.RandomState(seed)
    key = df[LABELS].astype(str).agg("".join, axis=1)
    out = []
    for k, g in df.groupby(key):
        m = int(round(n * len(g) / len(df)))
        out += list(rng.choice(g.index, size=min(m, len(g)), replace=False))
    out = out[:n]
    if len(out) < n:
        rest = df.index.difference(out)
        out += list(rng.choice(rest, size=n - len(out), replace=False))
    return df.loc[out]


def near_dup_filter(train, held, thr=0.85):
    """Buang dari train item yang near-duplicate (cosine char 3-5gram >= thr) dengan dev/test."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    vec.fit(pd.concat([train["text"], held["text"]]))
    A = vec.transform(train["text"])
    B = vec.transform(held["text"])
    sim = (A @ B.T).max(axis=1).toarray().ravel()
    mask = sim >= thr
    return train[~mask], int(mask.sum())


def build_oos(prefix, texts):
    rows = []
    for i, t in enumerate(texts):
        r = {"id": f"{prefix}{i:03d}", "raw": t, "text": t, "in_scope": False, "n_labels": 0}
        for l in LABELS:
            r[l] = 0
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Guideline: parsing + 3 strategi chunking (A1)
def parse_guideline(path):
    txt = open(path, encoding="utf-8").read()
    m = re.match(r"---\n(.*?)\n---\n(.*)", txt, re.S)
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    return meta, m.group(2)


def sections(body):
    """Pecah per heading level-3 (###) atau level-2 jika tak punya sub-heading. Return list (section_id, title, text, start, end)."""
    out = []
    heads = [(mm.start(), mm.group(1), mm.group(2).strip()) for mm in re.finditer(r"^(#{2,3}) (.+)$", body, re.M)]
    for i, (pos, lvl, title) in enumerate(heads):
        # section berakhir di heading berikutnya dengan level sama/lebih tinggi
        nxt = [h[0] for h in heads[i + 1:] if len(h[1]) <= len(lvl)]
        end = nxt[0] if nxt else len(body)
        out.append({"title": title, "level": len(lvl), "start": pos, "end": end, "text": body[pos:end].strip()})
    return out


def section_key(title):
    """Map judul bagian ke kunci qrels."""
    t = title.lower()
    if "ruang lingkup" in t or "luar cakupan" in t or "abstain" in t:
        return "SCOPE"
    if "hs_individual" in t or "target individu" in t:
        return "HS_Individual"
    if "hs_group" in t or "target kelompok" in t:
        return "HS_Group"
    if "abusive" in t or "bahasa kasar" in t:
        return "Abusive"
    if "hs" in t.split() or "hate speech" in t or "label hs" in t or "konsistensi hs" in t or "bukan hs" in t:
        return "HS"
    if "netral" in t or "kritik" in t:
        return "NONE"
    return "OTHER"


def chunk_guideline(meta, body, strategy):
    secs = sections(body)
    # tiap posisi karakter -> section key (pakai section terdalam)
    def key_at(a, b):
        keys = set()
        for s in secs:
            if s["start"] < b and a < s["end"]:
                k = section_key(s["title"])
                if s["level"] == 3 or not any(x["level"] == 3 and x["start"] >= s["start"] and x["end"] <= s["end"] for x in secs):
                    keys.add(k)
        return sorted(keys)

    chunks = []
    base = {"doc_type": "guideline", "version": meta["version"], "valid_from": meta["valid_from"],
            "valid_to": meta["valid_to"], "source": f"label_policy_v{meta['version']}.md"}
    if strategy == "fixed":
        size, step = 450, 350
        for i, a in enumerate(range(0, len(body), step)):
            b = min(a + size, len(body))
            chunks.append({**base, "chunk": f"fixed{i}", "text": body[a:b].strip(), "parent_text": None,
                           "keys": key_at(a, b)})
            if b == len(body):
                break
    elif strategy == "heading":
        for i, s in enumerate(secs):
            if s["level"] == 2 and any(x["level"] == 3 and x["start"] > s["start"] and x["end"] <= s["end"] for x in secs):
                # heading level-2 yang punya anak: hanya judul+intro sebelum sub-heading pertama
                first = min(x["start"] for x in secs if x["level"] == 3 and x["start"] > s["start"] and x["end"] <= s["end"])
                txt = body[s["start"]:first].strip()
                if len(txt) < 60:
                    continue
                chunks.append({**base, "chunk": f"head{i}", "text": txt, "parent_text": None, "keys": [section_key(s["title"])]})
            else:
                parent = next((x for x in secs if x["level"] == 2 and x["start"] <= s["start"] and x["end"] >= s["end"]), s)
                title = s["title"] if parent is s else parent["title"] + " > " + s["title"]
                chunks.append({**base, "chunk": f"head{i}", "text": s["text"], "parent_text": None,
                               "keys": [section_key(s["title"]) if section_key(s["title"]) != "OTHER" else section_key(parent["title"])],
                               "title": title})
    elif strategy == "hierarchical":
        # child = paragraf/kalimat-kelompok di dalam section level-3; parent = section level-2 utuh
        for i, s in enumerate(secs):
            has_child = any(x["level"] == 3 and x["start"] > s["start"] and x["end"] <= s["end"] for x in secs)
            if s["level"] == 2 and has_child:
                continue
            parent = next((x for x in secs if x["level"] == 2 and x["start"] <= s["start"] and x["end"] >= s["end"]), s)
            paras = [p.strip() for p in re.split(r"\n\s*\n|(?<=\.) (?=[A-Z\"])", s["text"]) if len(p.strip()) > 20]
            for j, p in enumerate(paras):
                k = section_key(s["title"])
                if k == "OTHER":
                    k = section_key(parent["title"])
                chunks.append({**base, "chunk": f"hier{i}_{j}", "text": p, "parent_text": parent["text"], "keys": [k]})
    for c in chunks:
        c["doc_id"] = f"G{meta['version']}_{c['chunk']}"
    return chunks


def guideline_keys(row):
    """Qrels guideline: bagian label positif; teks netral -> NONE; out-of-scope -> SCOPE."""
    if not row.get("in_scope", True):
        return ["SCOPE"]
    ks = [l for l in LABELS if row[l] == 1]
    return ks if ks else ["NONE"]


def build_lexicon(train, topn=60):
    """Lexicon chi2 per label dari TRAIN saja (dipakai untuk silver rationale & verifier)."""
    cv = CountVectorizer(token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", lowercase=True, min_df=5, binary=True)
    X = cv.fit_transform(train["text"])
    vocab = np.array(cv.get_feature_names_out())
    lex = {}
    stop = set("yang dan ini itu dengan untuk dari tidak ada akan juga user url karena sudah bisa saja kalau atau sama lagi jadi".split())
    for l in LABELS:
        y = train[l].values
        sc, _ = chi2(X, y)
        pos_rate = np.asarray(X[y == 1].mean(axis=0)).ravel()
        neg_rate = np.asarray(X[y == 0].mean(axis=0)).ravel()
        order = np.argsort(-np.nan_to_num(sc))
        words = [w for w in vocab[order] if w not in stop and pos_rate[list(vocab).index(w)] > neg_rate[list(vocab).index(w)]][:topn]
        lex[l] = words
    return lex


def main():
    df, stats = load_main()
    rng = np.random.RandomState(SEED)
    df["in_scope"] = True
    test = stratified_sample(df, N_TEST_INSCOPE, SEED)
    rest = df.drop(test.index)
    dev = stratified_sample(rest, N_DEV_INSCOPE, SEED + 1)
    train = rest.drop(dev.index)
    held = pd.concat([test, dev])
    train, n_near = near_dup_filter(train, held)
    stats["near_dup_removed_from_train"] = n_near
    # OOS: test = 20 teks BARU (tidak pernah dipakai menyetel scope_check); dev = 30 teks lama (set tuning)
    test = pd.concat([test, build_oos("OOS_T", OOS_TEXTS_TEST)], ignore_index=True)
    dev = pd.concat([dev, build_oos("OOS_D", OOS_TEXTS)], ignore_index=True)

    cols = ["id", "text", "raw", "in_scope"] + LABELS
    for name, d in [("train", train), ("dev", dev), ("test", test)]:
        d = d[cols].copy()
        d["split"] = name
        d.to_json(PROC / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)

    # Leakage check: tidak ada teks test/dev yang sama persis dengan train
    tr_keys = set(train["text"].map(norm_key))
    leak = int(sum(norm_key(t) in tr_keys for t in pd.concat([test, dev])["text"]))
    stats["exact_overlap_train_vs_heldout"] = leak

    # --- Corpus: contoh train (doc_type=example) + guideline chunks (3 strategi) ---
    examples = []
    for r in train.itertuples():
        examples.append({"doc_id": r.id, "doc_type": "example", "split": "train", "text": r.text,
                         "labels": {l: int(getattr(r, l)) for l in LABELS},
                         "source": "re_dataset.csv (Ibrohim & Budi 2019)", "version": "2019",
                         "valid_from": "2019-01-01", "valid_to": "9999-12-31"})
    with open(PROC / "corpus_examples.jsonl", "w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    chunk_stats = {}
    for strat in ["fixed", "heading", "hierarchical"]:
        allc = []
        for gp in sorted(GUIDE_DIR.glob("label_policy_v*.md")):
            meta, body = parse_guideline(gp)
            allc += chunk_guideline(meta, body, strat)
        with open(PROC / f"guide_chunks_{strat}.jsonl", "w", encoding="utf-8") as f:
            for c in allc:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        chunk_stats[strat] = {"n_chunks": len(allc), "total_chars": sum(len(c["text"]) for c in allc)}

    # --- Qrels (guideline-section level), untuk semua query dev/test ---
    qrels = []
    for name, d in [("dev", dev), ("test", test)]:
        for r in d.to_dict("records"):
            for k in guideline_keys(r):
                qrels.append({"qrel_id": f"Q_{r['id']}_{k}", "query_id": r["id"], "split": name,
                              "section_key": k, "relevance": 2 if k in LABELS or k == "SCOPE" else 1,
                              "doc_version": "1.0", "answerable": bool(r["in_scope"]),
                              "annotation_rule": "R1: section label positif (rel=2); netral->NONE (rel=1); out-of-scope->SCOPE (rel=2)"})
    with open(PROC / "qrels.jsonl", "w", encoding="utf-8") as f:
        for q in qrels:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    lex = build_lexicon(train)
    abusive_lex = [w.strip() for w in open(RAW / "abusive.csv", encoding="latin-1").read().splitlines()[1:] if w.strip()]
    json.dump({"chi2": lex, "abusive_csv": abusive_lex}, open(PROC / "lexicon.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # --- Manifest split + source register ---
    manifest = {"seed": SEED, "stats": stats, "chunk_stats": chunk_stats, "splits": {}}
    for name in ["train", "dev", "test"]:
        d = pd.read_json(PROC / f"{name}.jsonl", lines=True)
        manifest["splits"][name] = {
            "file": f"data/processed/{name}.jsonl", "sha256": sha256_file(PROC / f"{name}.jsonl"),
            "n": len(d), "n_in_scope": int(d["in_scope"].sum()), "n_out_of_scope": int((~d["in_scope"]).sum()),
            "label_prevalence": {l: round(float(d.loc[d.in_scope, l].mean()), 4) for l in LABELS}}
    manifest["corpus"] = {"n_example_docs": len(examples),
                          "corpus_examples_sha256": sha256_file(PROC / "corpus_examples.jsonl"),
                          "qrels_sha256": sha256_file(PROC / "qrels.jsonl"), "n_qrels": len(qrels)}
    manifest["locked_test"] = "test.jsonl dikunci: hanya dipakai untuk evaluasi akhir; pemilihan konfigurasi memakai dev."
    json.dump(manifest, open(PROC / "split_manifest.json", "w"), indent=2)

    register = [
        {"source_id": "S1", "name": "Indonesian multi-label hate speech & abusive (re_dataset.csv)",
         "url": "https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection",
         "license": "CC BY-NC-SA 4.0", "citation": "Ibrohim & Budi (2019), ALW3 ACL",
         "pii": "username & URL sudah dianonimkan (USER/URL) oleh penyedia", "sha256": sha256_file(RAW / "re_dataset.csv"),
         "use": "train/dev/test + corpus contoh"},
        {"source_id": "S2", "name": "new_kamusalay.csv (kamus normalisasi slang)", "url": "repo S1", "license": "CC BY-NC-SA 4.0",
         "sha256": sha256_file(RAW / "new_kamusalay.csv"), "use": "query planner / normalisasi"},
        {"source_id": "S3", "name": "abusive.csv (lexicon kata kasar)", "url": "repo S1", "license": "CC BY-NC-SA 4.0",
         "sha256": sha256_file(RAW / "abusive.csv"), "use": "tool lexicon_lookup / verifier"},
        {"source_id": "S4", "name": "IDHSD_RIO (Alfina et al. 2017) - OOD test", "url": "https://github.com/ialfina/id-hatespeech-detection",
         "license": "Bebas dipakai dengan sitasi; DILARANG redistribusi publik -> file tidak boleh di-commit ke repo publik",
         "sha256": sha256_file(DATA / "ood" / "IDHSD_RIO_unbalanced_713_2017.txt"), "use": "evaluasi OOD label HS (A4)"},
    ]
    for gp in sorted(GUIDE_DIR.glob("label_policy_v*.md")):
        meta, _ = parse_guideline(gp)
        register.append({"source_id": f"G{meta['version']}", "name": gp.name, "license": meta["license"], "author": meta["author"],
                         "valid_from": meta["valid_from"], "valid_to": meta["valid_to"], "sha256": sha256_file(gp),
                         "use": "label policy corpus (retrieval)"})
    json.dump(register, open(DATA / "source_register.json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
