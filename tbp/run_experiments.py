"""Runner eksperimen: retrieval ablation (A1,A2), dev selection, B0-B6 di locked test, A3-A7, repeatability, OOD.
Semua raw output per item + run manifest disimpan di results/runs/.  Jalankan: python -m tbp.run_experiments [stage...]
Stage: retrieval dev test ablations repeat ood  (default: semua)
"""
import collections
import hashlib
import json
import platform
import re
import sys
import time

import numpy as np
import pandas as pd

from .agent import AgentPipeline
from .config import (BUDGET_LARGE, BUDGET_SMALL, DATA, LABELS, PROC, REPEAT_SEEDS, REPEAT_SUBSET, RESULTS, RUNS, SEED)
from .data_prep import guideline_keys
from .evaluate import bootstrap_delta, evaluate_run, retrieval_metrics
from .pipelines import Pipeline
from .retrieval import get_index, load_jsonl
from .textutil import clean_text


def load(split):
    return load_jsonl(PROC / f"{split}.jsonl")


def save_run(run_id, recs, items, extra=None):
    p = RUNS / f"{run_id}.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = evaluate_run(items, recs)
    man = {"run_id": run_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_items": len(recs),
           "items_sha256": hashlib.sha256(json.dumps([i["id"] for i in items]).encode()).hexdigest()[:16],
           "output_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16],
           "meta": {k: recs[0].get(k) for k in ("config", "prompt_id", "prompt_hash", "adapter", "budget", "model_id",
                                                   "temperature", "seed", "as_of", "parent", "use_tools")},
           "index_versions": {n: get_index(n).version for n in ("examples", "guide_heading", "guide_fixed")},
           "env": {"python": sys.version.split()[0], "platform": platform.platform()}, "metrics": metrics}
    if extra:
        man.update(extra)
    json.dump(man, open(RUNS / f"{run_id}.manifest.json", "w"), indent=1)
    print(f"[{run_id}] macroF1={metrics['macro_f1']} microF1={metrics['micro_f1']} evid={metrics['evidence_support']} "
          f"oos_abst={metrics['oos_abstain_recall']} p95={metrics['cost']['latency_p95_ms']}", flush=True)
    return metrics


def load_run(run_id):
    return load_jsonl(RUNS / f"{run_id}.jsonl")


def results():
    p = RESULTS / "results.json"
    return json.load(open(p)) if p.exists() else {}


def put(key, val):
    r = results(); r[key] = val
    json.dump(r, open(RESULTS / "results.json", "w"), indent=1)


# ---------------------------------------------------------------- A1 + A2 (retrieval saja, tanpa LLM)
def stage_retrieval():
    test = load("test")
    out = {"A1_chunking": {}, "A2_retriever": {}}
    for strat in ["fixed", "heading", "hierarchical"]:
        idx = get_index(f"guide_{strat}")
        rows = []
        for it in test:
            rel = {k: (2 if k in LABELS or k == "SCOPE" else 1) for k in guideline_keys(it)}
            docs, ms = idx.search(it["text"] or " ", k=10, mode="hybrid_rerank", flt={"as_of": "2026-11-20"})
            m = retrieval_metrics([d["keys"] for d in docs], rel)
            ctx = sum(len(d.get("parent_text") or d["text"]) for d in docs[:5])
            # groundedness proxy: proporsi top-5 chunk yang relevan terhadap qrels
            m["top5_relevant_frac"] = float(np.mean([bool(set(d["keys"]) & set(rel)) for d in docs[:5]]))
            m["ctx_chars_top5"] = ctx; m["ms"] = ms
            rows.append(m)
        out["A1_chunking"][strat] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
        out["A1_chunking"][strat]["n_chunks"] = len(idx.docs)
        out["A1_chunking"][strat]["index_chars"] = sum(len(d["text"]) for d in idx.docs)
    idx = get_index("guide_heading")
    for mode in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
        rows = []
        for it in test:
            rel = {k: (2 if k in LABELS or k == "SCOPE" else 1) for k in guideline_keys(it)}
            docs, ms = idx.search(it["text"] or " ", k=10, mode=mode, flt={"as_of": "2026-11-20"})
            m = retrieval_metrics([d["keys"] for d in docs], rel); m["ms"] = ms
            rows.append(m)
        res = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
        res["p95_ms"] = round(float(np.percentile([r["ms"] for r in rows], 95)), 2)
        # neighbor label precision@5 pada index contoh (label tetangga == label gold)
        ex = get_index("examples")
        prec = []
        for it in [t for t in test if t["in_scope"]]:
            d5, _ = ex.search(it["text"], k=5, mode=mode, flt={"doc_type": "example"})
            prec.append(np.mean([all(d["labels"][l] == it[l] for l in LABELS) for d in d5]))
        res["example_label_precision@5"] = round(float(np.mean(prec)), 4)
        out["A2_retriever"][mode] = res
    print(json.dumps(out, indent=1))
    put("retrieval", out)


# ---------------------------------------------------------------- dev selection (parent B6)
def stage_dev():
    dev = load("dev")
    m4 = save_run("dev_B4", Pipeline("B4").run(dev), dev)
    m5 = save_run("dev_B5", Pipeline("B5").run(dev), dev)
    parent = "B5" if m5["macro_f1"] >= m4["macro_f1"] else "B4"
    put("dev_selection", {"B4_dev_macro_f1": m4["macro_f1"], "B5_dev_macro_f1": m5["macro_f1"], "B6_parent": parent,
                          "rule": "parent B6 = konfigurasi dengan macro-F1 dev tertinggi (B4 vs B5)"})
    return parent


def parent():
    return results().get("dev_selection", {}).get("B6_parent", "B4")


# ---------------------------------------------------------------- main test B0-B6
def stage_test():
    test = load("test")
    allm = {}
    for b in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        allm[b] = save_run(f"test_{b}", Pipeline(b).run(test), test)
    allm["B6"] = save_run("test_B6", AgentPipeline(parent(), use_tools=True).run(test), test)
    put("main_test", allm)
    deltas = {}
    pairs = [("B0", "B1"), ("B1", "B2"), ("B2", "B3"), ("B3", "B4"), ("B4", "B5"), (parent(), "B6"), ("B0", "B6")]
    for a, b in pairs:
        deltas[f"{b}-{a}"] = bootstrap_delta(test, load_run(f"test_{a}"), load_run(f"test_{b}"))
    put("deltas_bootstrap", deltas)


# ---------------------------------------------------------------- ablations
def stage_ablations():
    test = load("test")
    ab = {}
    # A3: no-verifier vs verifier+repair (pada B4 & B5, pakai pred sebelum verifikasi dari run yang sama)
    for b in ["B4", "B5"]:
        recs = load_run(f"test_{b}")
        nov = []
        for r in recs:
            r2 = dict(r)
            if "pred_before_verify" in r:
                r2["pred"] = r["pred_before_verify"]; r2["evidence"] = r["evidence_before_verify"]
                if r.get("abstain_reason") == "conflicting_evidence":
                    r2["abstain"], r2["abstain_reason"] = False, None
            nov.append(r2)
        m_nov = evaluate_run(test, nov); m_ver = evaluate_run(test, recs)
        ab[f"A3_{b}"] = {"no_verifier": {k: m_nov[k] for k in ("macro_f1", "unsupported_claim_rate", "evidence_support", "false_refusal_rate")},
                         "verifier_repair": {k: m_ver[k] for k in ("macro_f1", "unsupported_claim_rate", "evidence_support", "false_refusal_rate")},
                         "n_repair_calls": int(sum(r["n_llm_calls"] - 1 for r in recs if r["n_llm_calls"] > 1))}
    # A4: no-adapter vs LoRA pada parent B4 (lihat juga OOD + training log)
    m4, m5 = evaluate_run(test, load_run("test_B4")), evaluate_run(test, load_run("test_B5"))
    tl = json.load(open(DATA.parent / "adapters" / "lora_b5_training_log.json"))
    ab["A4"] = {"B4_no_adapter": {"macro_f1": m4["macro_f1"], "per_label": m4["per_label"]},
                "B5_lora": {"macro_f1": m5["macro_f1"], "per_label": m5["per_label"]},
                "delta": bootstrap_delta(test, load_run("test_B4"), load_run("test_B5")),
                "overfit_check": {"eval_curve": tl["eval"], "final_train_loss_last10": float(np.mean([s["train_loss"] for s in tl["steps"][-10:]]))},
                "resources": tl["resources"], "lora_config": tl["config"]}
    # A5: B6 tanpa vs dengan tool calling (parent identik)
    p = parent()
    m_nt = save_run("test_B6_notools", AgentPipeline(p, use_tools=False).run(test), test)
    recs6 = load_run("test_B6")
    ts = collections.Counter()
    for r in recs6:
        for k, v in (r.get("tool_stats") or {}).items():
            ts[k] += v
    att = max(1, ts["attempted"])
    ab["A5"] = {"parent": p, "without_tools": {"macro_f1": m_nt["macro_f1"], "cost": m_nt["cost"]},
                "with_tools": {"macro_f1": evaluate_run(test, recs6)["macro_f1"], "cost": evaluate_run(test, recs6)["cost"]},
                "tool_correctness": round(ts["ok"] / att, 4), "unsafe_call_rate": round((ts["unauthorized"] + ts["unsafe_args"]) / att, 4),
                "invalid_arg_rate": round(ts["invalid_args"] / att, 4), "budget_overrun_rate": round(ts["budget_overrun"] / len(recs6), 4),
                "parse_fail_steps": ts["parse_fail"], "tool_stats_total": dict(ts),
                "delta": bootstrap_delta(test, load_run("test_B6_notools"), recs6)}
    # A6: fixed vs adaptive retrieval policy (B4 tanpa retry vs B4) per slice
    save_run("test_B4_fixedretr", Pipeline("B4", overrides={"adaptive": False}).run(test), test)
    slices = make_slices(test)
    ab["A6"] = {}
    for run in ["test_B4_fixedretr", "test_B4"]:
        recs = load_run(run); ab["A6"][run] = {}
        for sl, ids in slices.items():
            its = [t for t in test if t["id"] in ids]
            rs = [r for r in recs if r["id"] in ids]
            m = evaluate_run(its, rs)
            ab["A6"][run][sl] = {"n": len(its), "macro_f1": m["macro_f1"] if sl != "no_evidence" else None,
                                 "exact_match": m["exact_match"] if sl != "no_evidence" else None,
                                 "oos_abstain_recall": m["oos_abstain_recall"], "mean_retries": m["cost"]["mean_retries"],
                                 "latency_mean_ms": m["cost"]["latency_mean_ms"], "mean_prompt_tokens": m["cost"]["mean_prompt_tokens"]}
    # A7: budget kecil vs besar
    m_large = save_run("test_B4_budgetlarge", Pipeline("B4", budget=BUDGET_LARGE).run(test), test)
    ab["A7"] = {"small": {"budget": BUDGET_SMALL, "macro_f1": m4["macro_f1"], "cost": m4["cost"]},
                "large": {"budget": BUDGET_LARGE, "macro_f1": m_large["macro_f1"], "cost": m_large["cost"]},
                "delta": bootstrap_delta(test, load_run("test_B4"), load_run("test_B4_budgetlarge"))}
    # A1 (level LLM): groundedness per strategi chunking pedoman pada B4 (heading = run B4 utama)
    ab["A1_llm"] = {}
    for strat in ("fixed", "heading", "hierarchical"):
        run_id = "test_B4" if strat == "heading" else f"test_B4_chunk_{strat}"
        m = (evaluate_run(test, load_run(run_id)) if strat == "heading"
             else save_run(run_id, Pipeline("B4", guide_strategy=strat).run(test), test))
        ab["A1_llm"][strat] = {"macro_f1": m["macro_f1"], "evidence_support": m["evidence_support"],
                               "unsupported_claim_rate": m["unsupported_claim_rate"],
                               "mean_prompt_tokens": m["cost"]["mean_prompt_tokens"], "latency_p95_ms": m["cost"]["latency_p95_ms"]}
    # B3 satu variabel: B3 dengan chunk FIXED (sama seperti B2) -> beda B2 vs B3 hanya metode retrieval
    m3f = save_run("test_B3_fixedchunk", Pipeline("B3", guide_strategy="fixed").run(test), test)
    ab["B3_single_variable"] = {"B3_fixedchunk_macro_f1": m3f["macro_f1"], "cost": m3f["cost"],
                                "delta_vs_B2": bootstrap_delta(test, load_run("test_B2"), load_run("test_B3_fixedchunk")),
                                "note": "B2 dan B3_fixedchunk sama-sama memakai chunk fixed; yang berbeda hanya retrieval (dense vs hybrid+rerank+filter)"}
    put("ablations", ab)


def make_slices(test):
    """Slice evaluasi (definisi dari gold + corpus train, bukan dari prediksi):
    easy = 0-1 label; multi = >=2 label; conflict = mayoritas label 5 tetangga BM25 != gold; no_evidence = out-of-scope."""
    ex = get_index("examples")
    s = {"easy": set(), "multi": set(), "conflict": set(), "no_evidence": set()}
    for it in test:
        if not it["in_scope"]:
            s["no_evidence"].add(it["id"]); continue
        n = sum(it[l] for l in LABELS)
        s["easy" if n <= 1 else "multi"].add(it["id"])
        d5, _ = ex.search(it["text"], k=5, mode="bm25", flt={"doc_type": "example"})
        maj = collections.Counter(tuple(d["labels"][l] for l in LABELS) for d in d5).most_common(1)[0][0]
        if maj != tuple(it[l] for l in LABELS):
            s["conflict"].add(it["id"])
    json.dump({k: sorted(v) for k, v in s.items()}, open(RESULTS / "slices.json", "w"), indent=1)
    return s


# ---------------------------------------------------------------- repeatability
def stage_repeat():
    test = load("test")
    rng = np.random.RandomState(SEED)
    ins = [t for t in test if t["in_scope"]]; oos = [t for t in test if not t["in_scope"]]
    sub = [ins[i] for i in rng.choice(len(ins), REPEAT_SUBSET - 5, replace=False)] + oos[:5]
    json.dump([s["id"] for s in sub], open(RESULTS / "repeat_subset_ids.json", "w"))
    rep = {}
    for b in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:
        vals = []
        for sd in REPEAT_SEEDS:
            pipe = AgentPipeline(parent(), use_tools=True) if b == "B6" else Pipeline(b)
            m = save_run(f"repeat_{b}_seed{sd}", pipe.run(sub, temperature=0.7, seed=sd), sub)
            vals.append(m["macro_f1"])
        rep[b] = {"macro_f1_runs": vals, "mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals, ddof=1)), 4),
                  "sampling": {"temperature": 0.7, "top_p": 0.9, "seeds": REPEAT_SEEDS}, "subset_n": len(sub)}
    put("repeatability", rep)


# ---------------------------------------------------------------- OOD (A4)
def load_ood(n=200, dedup=True):
    raw = open(DATA / "ood" / "IDHSD_RIO_unbalanced_713_2017.txt", encoding="latin-1").read().replace("\r\n", "\n").replace("\r", "\n")
    rows = []
    for i, line in enumerate(raw.split("\n")[1:]):
        if "\t" not in line:
            continue
        lab, tw = line.split("\t", 1)
        tw = re.sub(r"@\w+", "USER", tw)              # privasi: anonimisasi mention
        tw = re.sub(r"http\S+", "URL", tw)
        rows.append({"id": f"OOD{i:04d}", "text": tw.strip(), "in_scope": True, "HS": int(lab.strip() == "HS")})
    df = pd.DataFrame(rows)
    # Anti-leakage: dataset Ibrohim & Budi (2019) ikut memuat tweet Alfina dkk. (2017) (lihat paper Ibrohim §3).
    # Buang tweet OOD yang sama/near-duplicate (cosine char 3-5gram >= 0,8) dengan SELURUH dataset utama.
    if dedup:
        from sklearn.feature_extraction.text import TfidfVectorizer
        main = pd.read_csv(DATA / "raw" / "re_dataset.csv", encoding="latin-1")["Tweet"].map(clean_text)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([main, df.text]))
        sim = (vec.transform(df.text) @ vec.transform(main).T).max(axis=1).toarray().ravel()
        df["max_sim_main"] = sim
        stats = {"alfina_total": int(len(df)), "overlap_removed": int((sim >= 0.8).sum()), "clean_pool": int((sim < 0.8).sum()),
                 "rule": "buang jika cosine char 3-5gram >= 0.8 terhadap tweet mana pun di re_dataset.csv"}
        json.dump(stats, open(RESULTS / "ood_overlap.json", "w"), indent=1)
        df = df[sim < 0.8]
    df = df.sample(n=min(n, len(df)), random_state=SEED)
    return df.to_dict("records")


def stage_ood():
    items = load_ood()
    out = {}
    from .evaluate import f1
    for b in ["B4", "B5"]:
        recs = Pipeline(b).run(items)
        with open(RUNS / f"ood_{b}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {r["id"]: r for r in recs}
        tp = sum(it["HS"] == 1 and by[it["id"]]["pred"]["HS"] == 1 and not by[it["id"]]["abstain"] for it in items)
        fp = sum(it["HS"] == 0 and by[it["id"]]["pred"]["HS"] == 1 and not by[it["id"]]["abstain"] for it in items)
        fn = sum(it["HS"] == 1 and (by[it["id"]]["pred"]["HS"] == 0 or by[it["id"]]["abstain"]) for it in items)
        P, R, F = f1(tp, fp, fn)
        # macro-F1 biner (HS vs non-HS)
        tn = len(items) - tp - fp - fn
        F0 = f1(tn, fn, fp)[2]
        out[b] = {"HS_P": round(P, 4), "HS_R": round(R, 4), "HS_F1": round(F, 4), "binary_macro_f1": round((F + F0) / 2, 4),
                  "abstain_rate": round(float(np.mean([r["abstain"] for r in recs])), 4), "n": len(items)}
        print(b, out[b], flush=True)
    put("ood", out)


if __name__ == "__main__":
    stages = sys.argv[1:] or ["retrieval", "dev", "test", "ablations", "repeat", "ood"]
    for s in stages:
        t0 = time.time()
        print(f"===== stage {s} =====", flush=True)
        globals()[f"stage_{s}"]()
        print(f"===== stage {s} selesai {time.time() - t0:.0f}s =====", flush=True)
