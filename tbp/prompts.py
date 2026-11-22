"""Prompt registry (versioned + hashed). Output contract SAMA untuk B0-B6."""
import hashlib
import json

import pandas as pd

from .config import LABELS, PROC

OUTPUT_CONTRACT = '{"HS":0,"Abusive":0,"HS_Individual":0,"HS_Group":0,"evidence":[],"abstain":false}'

LABEL_DEFS = """Definisi label (1 = ya, 0 = tidak; boleh lebih dari satu label):
- HS: ujaran kebencian, menyerang/merendahkan/menghina individu atau kelompok.
- Abusive: mengandung kata kasar/makian/cabul (misal anjing, bangsat, goblok, tolol, bego, kampret, tai).
- HS_Individual: ujaran kebencian yang targetnya satu orang (tokoh, USER, lu/kamu/dia).
- HS_Group: ujaran kebencian yang targetnya kelompok (agama, etnis, partai, pendukung, kaum).
Aturan: jika HS=1 maka tepat satu dari HS_Individual/HS_Group bernilai 1; jika HS=0 keduanya 0.
Kata kasar tanpa menyerang pihak lain -> Abusive=1, HS=0.
"evidence": daftar potongan kata yang PERSIS ada di teks sebagai bukti label positif.
"abstain": true hanya jika teks bukan bahasa Indonesia/gaul, kosong, kode, atau hanya simbol."""

REGISTRY = {
    "P0_zero_shot@1.0": {
        "system": "Kamu adalah pengklasifikasi teks ujaran kebencian dan bahasa kasar. Jawab HANYA dengan satu objek JSON "
                  "dengan format: " + OUTPUT_CONTRACT,
        "user": "Teks: {text}\nJSON:",
    },
    "P1_structured@1.0": {
        "system": "Kamu adalah anotator ujaran kebencian dan bahasa kasar untuk tweet Indonesia.\n" + LABEL_DEFS +
                  "\nJawab HANYA dengan satu objek JSON berformat: " + OUTPUT_CONTRACT,
        "user": "Teks: {text}\nJSON:",
    },
    "P2_rag@1.0": {
        "system": "Kamu adalah anotator ujaran kebencian dan bahasa kasar untuk tweet Indonesia.\n" + LABEL_DEFS +
                  "\nGunakan KONTEKS (pedoman label dan contoh berlabel yang mirip) sebagai bukti pendukung, tetapi putuskan "
                  "berdasarkan isi teks. Jawab HANYA dengan satu objek JSON berformat: " + OUTPUT_CONTRACT,
        "user": "KONTEKS:\n{context}\n\nTeks: {text}\nJSON:",
    },
    "P_repair@1.0": {
        "system": "Kamu memeriksa bukti. Untuk label positif berikut, salin potongan kata yang PERSIS ada di teks sebagai bukti. "
                  "Jika tidak ada bukti, kosongkan. Jawab HANYA JSON: {\"evidence\": [\"...\"]}",
        "user": "Teks: {text}\nLabel positif: {labels}\nJSON:",
    },
    "P_agent@1.0": {
        "system": "Kamu adalah agen klasifikasi ujaran kebencian. Kamu boleh memanggil SATU tool per langkah untuk mengumpulkan "
                  "bukti sebelum klasifikasi. Tool tersedia:\n"
                  "- search_examples(query: str, k: int<=10): cari contoh berlabel yang mirip.\n"
                  "- get_policy(label: one of HS|Abusive|HS_Individual|HS_Group|NONE|SCOPE, as_of: YYYY-MM-DD): ambil pedoman label.\n"
                  "- lexicon_lookup(word: str): cek apakah kata termasuk kata kasar/penanda ujaran kebencian.\n"
                  "- normalize_slang(text: str): ubah kata gaul ke baku.\n"
                  "Jawab HANYA JSON: {\"tool\": \"<nama>\", \"args\": {...}} atau {\"done\": true} jika bukti sudah cukup.",
        "user": "Teks: {text}\nTanggal hari ini (as_of): {as_of}\nObservasi sebelumnya:\n{obs}\nLangkah berikutnya (JSON):",
    },
}


def prompt_hash(pid):
    return hashlib.sha256(json.dumps(REGISTRY[pid], sort_keys=True).encode()).hexdigest()[:12]


def few_shot_examples(n=4, seed=7):
    """Few-shot TETAP dari TRAIN (tidak pernah dari test) mencakup kombinasi label berbeda."""
    tr = pd.read_json(PROC / "train.jsonl", lines=True)
    tr = tr[tr.text.str.len().between(30, 140)]
    combos = [(0, 0, 0, 0), (0, 1, 0, 0), (1, 1, 1, 0), (1, 0, 0, 1)]
    out = []
    for c in combos[:n]:
        g = tr[(tr.HS == c[0]) & (tr.Abusive == c[1]) & (tr.HS_Individual == c[2]) & (tr.HS_Group == c[3])]
        out.append(g.sample(1, random_state=seed).iloc[0])
    return out


def example_answer(row, evidence):
    d = {l: int(row[l]) for l in LABELS}
    d["evidence"] = evidence
    d["abstain"] = False
    return json.dumps(d, ensure_ascii=False)


def build_messages(pid, text, context=None, few_shot=None, **kw):
    p = REGISTRY[pid]
    msgs = [{"role": "system", "content": p["system"]}]
    for fs_text, fs_ans in (few_shot or []):
        u = p["user"].format(text=fs_text, context="(contoh)", **{k: "" for k in ("labels", "as_of", "obs")})
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": fs_ans}]
    msgs.append({"role": "user", "content": p["user"].format(text=text, context=context or "", **kw)})
    return msgs
