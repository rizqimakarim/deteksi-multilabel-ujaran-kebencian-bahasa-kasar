"""Human review (blind, 2 annotator + adjudikasi).

  1) python -m tbp.human_review make      -> buat human_review/sheet_annotator_A.csv & _B.csv (urutan acak berbeda,
                                              TANPA label gold & TANPA prediksi model = blind)
  2) Annotator A dan B mengisi kolom HS, Abusive, HS_Individual, HS_Group (0/1) secara terpisah sesuai
     data/guidelines/label_policy_v1.0.md
  3) python -m tbp.human_review agree     -> Cohen's kappa per label, % agreement, disagreement_log.csv,
                                              adjudication_sheet.csv (hanya item yang berbeda)
  4) Adjudikator mengisi adjudication_sheet.csv (kolom final_* + resolution_note)
  5) python -m tbp.human_review finalize  -> human_gold.csv + perbandingan dengan label dataset & evaluasi model
"""
import json
import sys

import numpy as np
import pandas as pd

from .config import LABELS, PROC, RESULTS, ROOT

HR = ROOT / "human_review"
HR.mkdir(exist_ok=True)
N = 100


def make():
    test = pd.read_json(PROC / "test.jsonl", lines=True)
    sub = test[test.in_scope].sample(N, random_state=2026)[["id", "text"]]
    for who, seed in (("A", 1), ("B", 2)):
        s = sub.sample(frac=1, random_state=seed).copy()
        for l in LABELS:
            s[l] = ""
        s["catatan"] = ""
        s.to_csv(HR / f"sheet_annotator_{who}.csv", index=False, encoding="utf-8-sig")
    json.dump({"n": N, "ids": sub["id"].tolist(), "guideline": "label_policy_v1.0.md", "blind": True,
               "instruksi": "Isi 0/1 per label tanpa melihat label dataset/prediksi model; jangan berdiskusi sebelum selesai."},
              open(HR / "protocol.json", "w"), indent=1)
    print("sheet dibuat di", HR)


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree():
    A = pd.read_csv(HR / "sheet_annotator_A.csv").set_index("id")
    B = pd.read_csv(HR / "sheet_annotator_B.csv").set_index("id").loc[A.index]
    if A[LABELS].isna().any().any() or B[LABELS].isna().any().any():
        sys.exit("Sheet belum lengkap diisi.")
    res = {l: {"kappa": round(kappa(A[l], B[l]), 4), "pct_agree": round(float((A[l] == B[l]).mean()), 4)} for l in LABELS}
    res["exact_vector_agree"] = round(float((A[LABELS] == B[LABELS]).all(axis=1).mean()), 4)
    dis = A[(A[LABELS] != B[LABELS]).any(axis=1)].copy()
    rows = []
    for i in dis.index:
        r = {"id": i, "text": A.loc[i, "text"]}
        for l in LABELS:
            r[f"A_{l}"], r[f"B_{l}"], r[f"final_{l}"] = A.loc[i, l], B.loc[i, l], ""
        r["resolution_note"] = ""
        rows.append(r)
    pd.DataFrame(rows).to_csv(HR / "adjudication_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows)[["id", "text"] + [c for c in rows[0] if c.startswith(("A_", "B_"))]].to_csv(
        HR / "disagreement_log.csv", index=False, encoding="utf-8-sig") if rows else None
    res["n_disagreements"] = len(rows)
    json.dump(res, open(HR / "agreement.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def finalize():
    A = pd.read_csv(HR / "sheet_annotator_A.csv").set_index("id")
    adj = pd.read_csv(HR / "adjudication_sheet.csv").set_index("id") if (HR / "adjudication_sheet.csv").exists() else None
    gold = A[["text"] + LABELS].copy()
    if adj is not None:
        for i in adj.index:
            for l in LABELS:
                gold.loc[i, l] = int(adj.loc[i, f"final_{l}"])
    gold.to_csv(HR / "human_gold.csv", encoding="utf-8-sig")
    test = pd.read_json(PROC / "test.jsonl", lines=True).set_index("id").loc[gold.index]
    agree_ds = {l: round(kappa(gold[l].astype(int), test[l]), 4) for l in LABELS}
    out = {"kappa_human_gold_vs_dataset_label": agree_ds}
    # evaluasi model terhadap human gold
    from .evaluate import label_metrics
    for b in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
        p = RESULTS / "runs" / f"test_{b}.jsonl"
        if p.exists():
            recs = {r["id"]: r for r in pd.read_json(p, lines=True).to_dict("records")}
            g = [{l: int(gold.loc[i, l]) for l in LABELS} for i in gold.index]
            pr = [({l: 0 for l in LABELS} if recs[i]["abstain"] else recs[i]["pred"]) for i in gold.index]
            out[b] = label_metrics(g, pr)["macro_f1"]
    json.dump(out, open(HR / "human_gold_eval.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"make": make, "agree": agree, "finalize": finalize}[sys.argv[1]]()
