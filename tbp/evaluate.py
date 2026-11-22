"""Metrik: multi-label F1, abstain, evidence support, kalibrasi (ECE), biaya, retrieval (Recall/nDCG/MRR), bootstrap CI."""
import math

import numpy as np

from .config import LABELS
from .textutil import span_in_text


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def label_metrics(golds, preds):
    out = {}
    TP = FP = FN = 0
    for l in LABELS:
        tp = sum(g[l] == 1 and p[l] == 1 for g, p in zip(golds, preds))
        fp = sum(g[l] == 0 and p[l] == 1 for g, p in zip(golds, preds))
        fn = sum(g[l] == 1 and p[l] == 0 for g, p in zip(golds, preds))
        P, R, F = f1(tp, fp, fn)
        out[l] = {"P": round(P, 4), "R": round(R, 4), "F1": round(F, 4), "support": tp + fn}
        TP += tp; FP += fp; FN += fn
    macro = float(np.mean([out[l]["F1"] for l in LABELS]))
    micro = f1(TP, FP, FN)[2]
    exact = float(np.mean([all(g[l] == p[l] for l in LABELS) for g, p in zip(golds, preds)])) if golds else 0.0
    hamming = float(np.mean([sum(g[l] != p[l] for l in LABELS) / len(LABELS) for g, p in zip(golds, preds)])) if golds else 0.0
    return {"macro_f1": round(macro, 4), "micro_f1": round(micro, 4), "exact_match": round(exact, 4),
            "hamming_loss": round(hamming, 4), "per_label": out}


def ece(golds, recs, bins=10):
    ys, ps = [], []
    for g, r in zip(golds, recs):
        for l in LABELS:
            ys.append(g[l]); ps.append(r["scores"][l])
    ys, ps = np.array(ys), np.array(ps)
    conf = np.where(ps >= 0.5, ps, 1 - ps)
    pred = (ps >= 0.5).astype(int)
    e = 0.0
    for b in range(bins):
        lo, hi = 0.5 + b * 0.05, 0.5 + (b + 1) * 0.05
        m = (conf >= lo) & ((conf < hi) if b < bins - 1 else (conf <= hi))
        if m.any():
            e += m.mean() * abs((pred[m] == ys[m]).mean() - conf[m].mean())
    return round(float(e), 4)


def evaluate_run(items, recs):
    """items: list gold dict (dengan in_scope); recs: list output pipeline (urutan sama)."""
    by = {r["id"]: r for r in recs}
    ins = [it for it in items if it["in_scope"]]
    oos = [it for it in items if not it["in_scope"]]
    g_in = [{l: it[l] for l in LABELS} for it in ins]
    p_in = []
    for it in ins:
        r = by[it["id"]]
        p_in.append({l: 0 for l in LABELS} if r["abstain"] else r["pred"])  # abstain -> negatif (konservatif)
    res = label_metrics(g_in, p_in)
    answered = [(g, by[it["id"]]["pred"]) for g, it in zip(g_in, ins) if not by[it["id"]]["abstain"]]
    res["selective_macro_f1"] = label_metrics([a for a, _ in answered], [b for _, b in answered])["macro_f1"] if answered else 0.0
    res["coverage_in_scope"] = round(len(answered) / max(1, len(ins)), 4)
    res["false_refusal_rate"] = round(1 - res["coverage_in_scope"], 4)
    res["oos_abstain_recall"] = round(float(np.mean([by[it["id"]]["abstain"] for it in oos])), 4) if oos else None
    # evidence support: item terjawab dengan >=1 label positif yang punya >=1 span verbatim
    pos_items = [(it, by[it["id"]]) for it in ins if not by[it["id"]]["abstain"] and any(by[it["id"]]["pred"].values())]
    sup = [any(span_in_text(e, it["text"]) for e in r["evidence"]) for it, r in pos_items]
    res["evidence_support"] = round(float(np.mean(sup)), 4) if sup else None
    n_pos = sum(sum(r["pred"].values()) for _, r in pos_items)
    n_unsup = sum(sum(r["pred"].values()) for (it, r), s in zip(pos_items, sup) if not s)
    res["unsupported_claim_rate"] = round(n_unsup / n_pos, 4) if n_pos else None
    res["parse_ok_rate"] = round(float(np.mean([by[it["id"]]["parse_ok"] for it in items])), 4)
    res["ece"] = ece(g_in, [by[it["id"]] for it in ins])
    lat = np.array([by[it["id"]]["latency_ms"] for it in items])
    res["cost"] = {"mean_prompt_tokens": round(float(np.mean([by[i["id"]]["prompt_tokens"] for i in items])), 1),
                   "mean_completion_tokens": round(float(np.mean([by[i["id"]]["completion_tokens"] for i in items])), 1),
                   "mean_llm_calls": round(float(np.mean([by[i["id"]]["n_llm_calls"] for i in items])), 3),
                   "latency_mean_ms": round(float(lat.mean()), 1), "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
                   "mean_retries": round(float(np.mean([by[i["id"]].get("retries", 0) for i in items])), 3)}
    return res


def bootstrap_delta(items, recs_a, recs_b, n=1000, seed=0):
    """Paired bootstrap 95% CI untuk delta macro-F1 (B - A) pada item in-scope."""
    ins = [it for it in items if it["in_scope"]]
    A = {r["id"]: r for r in recs_a}; B = {r["id"]: r for r in recs_b}
    def pred(r): return {l: 0 for l in LABELS} if r["abstain"] else r["pred"]
    g = [{l: it[l] for l in LABELS} for it in ins]
    pa = [pred(A[it["id"]]) for it in ins]; pb = [pred(B[it["id"]]) for it in ins]
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n):
        idx = rng.randint(0, len(ins), len(ins))
        ds.append(label_metrics([g[i] for i in idx], [pb[i] for i in idx])["macro_f1"] -
                  label_metrics([g[i] for i in idx], [pa[i] for i in idx])["macro_f1"])
    d = label_metrics(g, pb)["macro_f1"] - label_metrics(g, pa)["macro_f1"]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"delta": round(d, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "significant": bool(lo > 0 or hi < 0)}


# ---------------- retrieval metrics ----------------
def retrieval_metrics(ranked_keys, rel, ks=(5, 10)):
    """ranked_keys: list (per posisi) set kunci section yang dikandung chunk; rel: dict key->relevance.
    Item relevan = section key. Recall@k = proporsi key relevan yang tercakup di top-k."""
    out = {}
    for k in ks:
        cov = set()
        for ks_ in ranked_keys[:k]:
            cov |= (set(ks_) & set(rel))
        out[f"recall@{k}"] = len(cov) / len(rel)
    # nDCG@10 & MRR@10 (gain = relevansi tertinggi key baru yang tercakup chunk)
    dcg, seen, mrr = 0.0, set(), 0.0
    for i, ks_ in enumerate(ranked_keys[:10]):
        new = (set(ks_) & set(rel)) - seen
        gain = max([rel[x] for x in new], default=0)
        seen |= new
        dcg += (2 ** gain - 1) / math.log2(i + 2)
        if mrr == 0 and set(ks_) & set(rel):
            mrr = 1 / (i + 1)
    ideal = sorted(rel.values(), reverse=True)
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:10]))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    out["mrr@10"] = mrr
    return out
