"""Bangun tabel laporan (results/report_tables.md) dari results.json + raw runs: termasuk hallucination audit & fairness.
Jalankan: python -m tbp.make_report
"""
import collections
import json
import re

import numpy as np

from .config import LABELS, PROC, RESULTS, RUNS
from .llm import parse_output
from .retrieval import load_jsonl
from .textutil import span_in_text

IDENTITY = r"\b(islam|muslim|kristen|katolik|hindu|budha|buddha|cina|china|tionghoa|aseng|jawa|batak|arab|yahudi|perempuan|wanita|cewek|lgbt|gay|banci|kafir|ulama|ustad|pendeta)\b"


def tbl(headers, rows):
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |\n"
    return s


def hallucination_audit(run_id, items):
    """Span yang dikarang = span keluaran mentah LLM yang tidak ada verbatim di teks."""
    by = {i["id"]: i for i in items}
    n_sp = n_hal = n_items_hal = n_items = 0
    for r in load_jsonl(RUNS / f"{run_id}.jsonl"):
        if not r.get("raw_output"):
            continue
        p, _ = parse_output(r["raw_output"])
        sp = [e for e in p["evidence"] if str(e).strip()]
        if not sp:
            continue
        n_items += 1
        bad = [e for e in sp if not span_in_text(e, by[r["id"]]["text"])]
        n_sp += len(sp); n_hal += len(bad); n_items_hal += bool(bad)
    return {"raw_spans": n_sp, "hallucinated_span_rate": round(n_hal / n_sp, 4) if n_sp else None,
            "items_with_hallucinated_span": round(n_items_hal / n_items, 4) if n_items else None}


def fairness(run_id, items):
    """FPR label HS pada teks NON-HS in-scope yang menyebut identitas kelompok vs yang tidak."""
    by = {r["id"]: r for r in load_jsonl(RUNS / f"{run_id}.jsonl")}
    g = {"identity": [0, 0], "no_identity": [0, 0]}
    for it in items:
        if not it["in_scope"] or it["HS"] == 1:
            continue
        k = "identity" if re.search(IDENTITY, it["text"].lower()) else "no_identity"
        r = by[it["id"]]
        g[k][0] += int(r["pred"]["HS"] == 1 and not r["abstain"]); g[k][1] += 1
    return {k: {"fpr_HS": round(v[0] / v[1], 4) if v[1] else None, "n": v[1]} for k, v in g.items()}


CFG_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]


def error_analysis(test):
    """Taksonomi penyebab error (6 sumber di PDF §1: retrieval, reasoning, tool, source, prompt, adaptation).
    Urutan pemeriksaan (deterministik, per item in-scope dengan vektor label salah):
      format/prompt -> abstain -> source (dugaan noise label) -> tool (B6) -> retrieval -> reasoning/model | no-retrieval.
    adaptation = efek LoRA, dibaca dari perubahan kolom reasoning B4 -> B5."""
    lx = json.load(open(PROC / "lexicon.json", encoding="utf-8"))
    abusive = set(lx["abusive_csv"])
    ex = {d["doc_id"]: d["labels"] for d in load_jsonl(PROC / "corpus_examples.jsonl")}
    by_t = {t["id"]: t for t in test}
    cols = ["format", "abstain", "source", "tool", "retrieval", "reasoning", "noret"]
    rows, runs = [], {}
    for n in CFG_ORDER:
        p = RUNS / f"test_{n}.jsonl"
        if not p.exists():
            continue
        runs[n] = {r["id"]: r for r in load_jsonl(p)}
        c = dict.fromkeys(cols, 0)
        wrong = 0
        for i, r in runs[n].items():
            g = by_t[i]
            if not g["in_scope"]:
                continue
            gv = tuple(g[l] for l in LABELS)
            pv = (0,) * 4 if r["abstain"] else tuple(r["pred"][l] for l in LABELS)
            if pv == gv:
                continue
            wrong += 1
            toks = set(re.findall(r"[a-z]+", g["text"].lower()))
            if not r["parse_ok"]:
                c["format"] += 1
            elif r["abstain"]:
                c["abstain"] += 1
            elif g["Abusive"] == 0 and r["pred"]["Abusive"] == 1 and toks & abusive:
                c["source"] += 1          # gold bilang tidak kasar, padahal ada kata dari lexicon kasar penyedia dataset
            elif n == "B6" and (r.get("tool_stats") or {}).get("ok", 0) == 0:
                c["tool"] += 1            # tidak ada satu pun tool call yang valid -> agent gagal mengumpulkan bukti
            elif not r["retrieved_ids"]:
                c["noret"] += 1
            else:
                vecs = [tuple(ex[d][l] for l in LABELS) for d in r["retrieved_ids"] if d in ex]
                maj = collections.Counter(vecs).most_common(1)[0][0] if vecs else None
                c["retrieval" if maj != gv else "reasoning"] += 1
        rows.append([n, wrong] + [c[k] for k in cols])
    md = ("# Analisis Penyebab Error (locked test, item in-scope dengan vektor label salah)\n\n"
          "Taksonomi deterministik, diperiksa berurutan: **format/prompt** = output bukan JSON valid · **abstain** = in-scope di-abstain · "
          "**source** = dugaan noise label sumber (gold Abusive=0 padahal tweet memuat kata dari `abusive.csv` milik penyedia dataset, dan model memprediksi Abusive=1) · "
          "**tool** = (B6) tak satu pun tool call valid · **retrieval** = mayoritas label contoh ter-retrieve ≠ gold · "
          "**reasoning/model** = bukti retrieval mayoritas benar tetapi LLM salah · **no-retrieval** = konfigurasi tanpa retrieval (B0/B1). "
          "**adaptation** (efek LoRA) dibaca dari perubahan kolom reasoning B4 → B5.\n\n")
    md += tbl(["Config", "#salah", "format/prompt", "abstain", "source", "tool", "retrieval", "reasoning/model", "no-retrieval"], rows)
    # contoh kualitatif
    if all(k in runs for k in ("B4", "B5", "B6")):
        b4, b5, b6 = runs["B4"], runs["B5"], runs["B6"]
        ok = lambda r, g: (not r["abstain"]) and r["pred"] == {l: g[l] for l in LABELS}
        cases = {"B6 benar, B5 salah": [], "B5 benar, B4 salah": [], "Semua salah": []}
        for i, g in by_t.items():
            if not g["in_scope"]:
                continue
            if ok(b6[i], g) and not ok(b5[i], g): cases["B6 benar, B5 salah"].append(i)
            if ok(b5[i], g) and not ok(b4[i], g): cases["B5 benar, B4 salah"].append(i)
            if not ok(b4[i], g) and not ok(b5[i], g) and not ok(b6[i], g): cases["Semua salah"].append(i)
        md += "\n## Contoh kualitatif\n\n"
        for k_, ids in cases.items():
            md += f"### {k_} ({len(ids)} item) — 2 contoh\n\n"
            for i in ids[:2]:
                g = by_t[i]
                md += (f"- **{i}** `{g['text'][:160]}`  \n  gold={ {l: g[l] for l in LABELS} }  \n"
                       f"  B4={b4[i]['pred']} · B5={b5[i]['pred']} · B6={b6[i]['pred']} · span B6={b6[i]['evidence']}  \n"
                       f"  tool B6: {[(t.get('call', {}).get('tool'), t['status']) for t in (b6[i].get('tool_log') or [])]}\n\n")
    open(RESULTS / "error_analysis.md", "w", encoding="utf-8").write(md)
    return md


def main():
    R = json.load(open(RESULTS / "results.json"))
    test = load_jsonl(PROC / "test.jsonl")
    md = "# Tabel Hasil (auto-generated dari results/results.json)\n\n"
    if "main_test" in R:
        md += "## T1. Perbandingan B0-B6 pada locked test (200 in-scope + 20 out-of-scope)\n\n"
        rows = []
        for b, m in R["main_test"].items():
            rows.append([b, m["macro_f1"], m["micro_f1"], m["exact_match"], m["hamming_loss"], m["evidence_support"],
                         m["unsupported_claim_rate"], m["oos_abstain_recall"], m["false_refusal_rate"], m["ece"], m["parse_ok_rate"]])
        md += tbl(["Config", "Macro-F1", "Micro-F1", "Exact", "Hamming", "Evid.support", "Unsupp.claim", "OOS abstain", "False refusal", "ECE", "Parse OK"], rows)
        md += "\n## T2. Per-label F1\n\n"
        md += tbl(["Config"] + LABELS, [[b] + [m["per_label"][l]["F1"] for l in LABELS] for b, m in R["main_test"].items()])
        md += "\n## T3. Biaya per item\n\n"
        md += tbl(["Config", "Prompt tok", "Completion tok", "LLM calls", "Latency mean (ms)", "Latency p95 (ms)", "Retries"],
                  [[b, m["cost"]["mean_prompt_tokens"], m["cost"]["mean_completion_tokens"], m["cost"]["mean_llm_calls"],
                    m["cost"]["latency_mean_ms"], m["cost"]["latency_p95_ms"], m["cost"]["mean_retries"]] for b, m in R["main_test"].items()])
    if "deltas_bootstrap" in R:
        md += "\n## T4. Delta macro-F1 (paired bootstrap 1000x, CI 95%)\n\n"
        md += tbl(["Perbandingan", "Delta", "CI95 low", "CI95 high", "Signifikan"],
                  [[k, v["delta"], v["ci95"][0], v["ci95"][1], v["significant"]] for k, v in R["deltas_bootstrap"].items()])
    if "dev_selection" in R:
        d = R["dev_selection"]
        md += f"\n**Seleksi parent B6 (dev):** B4 dev macro-F1 = {d['B4_dev_macro_f1']}, B5 dev macro-F1 = {d['B5_dev_macro_f1']} -> parent = **{d['B6_parent']}**.\n"
    if "retrieval" in R:
        md += "\n## T5. A1 Chunking guideline (hybrid+rerank, qrels section-level)\n\n"
        a1 = R["retrieval"]["A1_chunking"]
        md += tbl(["Strategi", "Recall@5", "Recall@10", "nDCG@10", "MRR@10", "Top5 relevan", "Ctx chars top5", "#chunk", "Index chars"],
                  [[k, v["recall@5"], v["recall@10"], v["ndcg@10"], v["mrr@10"], v["top5_relevant_frac"], v["ctx_chars_top5"], v["n_chunks"], v["index_chars"]] for k, v in a1.items()])
        md += "\n## T6. A2 Retriever (heading-aware chunks, qrels identik)\n\n"
        a2 = R["retrieval"]["A2_retriever"]
        md += tbl(["Retriever", "Recall@5", "Recall@10", "nDCG@10", "MRR@10", "Label-precision@5 (contoh)", "Latency mean ms", "p95 ms"],
                  [[k, v["recall@5"], v["recall@10"], v["ndcg@10"], v["mrr@10"], v["example_label_precision@5"], v["ms"], v["p95_ms"]] for k, v in a2.items()])
    ab = R.get("ablations", {})
    if ab:
        md += "\n## T7. A3 Verifier\n\n"
        rows = []
        for b in ("B4", "B5"):
            a = ab.get(f"A3_{b}")
            if a:
                for k in ("no_verifier", "verifier_repair"):
                    rows.append([f"{b} {k}", a[k]["macro_f1"], a[k]["unsupported_claim_rate"], a[k]["evidence_support"], a[k]["false_refusal_rate"]])
        md += tbl(["Varian", "Macro-F1", "Unsupported claim", "Evidence support", "False refusal"], rows)
        a4 = ab["A4"]
        md += "\n## T8. A4 LoRA vs tanpa adapter (parent B4)\n\n"
        md += tbl(["Varian", "Macro-F1 test"] + LABELS,
                  [["B4 (no adapter)", a4["B4_no_adapter"]["macro_f1"]] + [a4["B4_no_adapter"]["per_label"][l]["F1"] for l in LABELS],
                   ["B5 (LoRA)", a4["B5_lora"]["macro_f1"]] + [a4["B5_lora"]["per_label"][l]["F1"] for l in LABELS]])
        md += f"\nDelta: {a4['delta']}. Resource: {a4['resources']}.\n\nKurva loss (overfit check): {a4['overfit_check']['eval_curve']}\n"
        if "ood" in R:
            md += "\n**OOD (Alfina et al. 2017, label HS saja, n=200):**\n\n"
            md += tbl(["Varian", "HS-P", "HS-R", "HS-F1", "Binary macro-F1", "Abstain"],
                      [[b, v["HS_P"], v["HS_R"], v["HS_F1"], v["binary_macro_f1"], v["abstain_rate"]] for b, v in R["ood"].items()])
        a5 = ab["A5"]
        md += f"\n## T9. A5 B6 tanpa vs dengan tool calling (parent {a5['parent']})\n\n"
        md += tbl(["Varian", "Macro-F1", "Latency mean ms", "LLM calls"],
                  [["tanpa tool", a5["without_tools"]["macro_f1"], a5["without_tools"]["cost"]["latency_mean_ms"], a5["without_tools"]["cost"]["mean_llm_calls"]],
                   ["dengan tool", a5["with_tools"]["macro_f1"], a5["with_tools"]["cost"]["latency_mean_ms"], a5["with_tools"]["cost"]["mean_llm_calls"]]])
        md += (f"\nTool correctness = {a5['tool_correctness']}, unsafe call rate = {a5['unsafe_call_rate']}, invalid-arg rate = {a5['invalid_arg_rate']}, "
               f"budget overrun rate (per item) = {a5['budget_overrun_rate']}, parse-fail steps = {a5['parse_fail_steps']}. Total: {a5['tool_stats_total']}. Delta: {a5['delta']}\n")
        md += "\n## T10. A6 Fixed vs adaptive retrieval per slice\n\n"
        rows = []
        for run, sl in ab["A6"].items():
            for k, v in sl.items():
                rows.append([run.replace("test_", ""), k, v["n"], v["macro_f1"], v["exact_match"], v["oos_abstain_recall"], v["mean_retries"], v["latency_mean_ms"], v["mean_prompt_tokens"]])
        md += tbl(["Run", "Slice", "n", "Macro-F1", "Exact", "OOS abstain", "Retries", "Latency ms", "Prompt tok"], rows)
        # biaya per kualitas (keseluruhan test)
        cq = []
        for run in ("test_B4_fixedretr", "test_B4"):
            p = RUNS / f"{run}.manifest.json"
            if p.exists():
                m = json.load(open(p))["metrics"]
                f1_pts = m["macro_f1"] * 100
                cq.append([run.replace("test_", ""), m["macro_f1"], m["cost"]["mean_prompt_tokens"], m["cost"]["latency_mean_ms"],
                           round(m["cost"]["mean_prompt_tokens"] / f1_pts, 2) if f1_pts else None,
                           round(m["cost"]["latency_mean_ms"] / f1_pts, 2) if f1_pts else None])
        if cq:
            md += "\n**Biaya per kualitas (seluruh test):**\n\n"
            md += tbl(["Run", "Macro-F1", "Prompt tok/item", "Latency ms/item", "Prompt tok per poin F1", "ms per poin F1"], cq)
        a7 = ab["A7"]
        md += "\n## T11. A7 Budget konteks\n\n"
        md += tbl(["Budget", "k contoh", "k pedoman", "Max ctx chars", "Macro-F1", "Prompt tok", "Latency mean", "p95"],
                  [[n, a7[n]["budget"]["k_examples"], a7[n]["budget"]["k_guides"], a7[n]["budget"]["max_ctx_chars"], a7[n]["macro_f1"],
                    a7[n]["cost"]["mean_prompt_tokens"], a7[n]["cost"]["latency_mean_ms"], a7[n]["cost"]["latency_p95_ms"]] for n in ("small", "large")])
        md += f"\nDelta: {a7['delta']}\n"
        if "A1_llm" in ab:
            md += "\n## T11b. A1 level LLM — groundedness per strategi chunking (B4, locked test)\n\n"
            md += tbl(["Strategi", "Macro-F1", "Evidence support", "Unsupported claim", "Prompt tok", "Latency p95"],
                      [[k_, v["macro_f1"], v["evidence_support"], v["unsupported_claim_rate"], v["mean_prompt_tokens"], v["latency_p95_ms"]]
                       for k_, v in ab["A1_llm"].items()])
        if "B3_single_variable" in ab:
            b3 = ab["B3_single_variable"]
            md += ("\n## T11c. B3 satu variabel (chunk fixed seperti B2)\n\n"
                   f"B3 dengan chunk fixed: macro-F1 {b3['B3_fixedchunk_macro_f1']}; delta vs B2 = {b3['delta_vs_B2']}. {b3['note']}\n")
    if "repeatability" in R:
        md += "\n## T12. Repeatability (3 run independen, sampling T=0.7, subset 40)\n\n"
        md += tbl(["Config", "Run1", "Run2", "Run3", "Mean", "Std"],
                  [[b, *v["macro_f1_runs"], v["mean"], v["std"]] for b, v in R["repeatability"].items()])
    if "main_test" in R:
        md += "\n## T13. Hallucination audit (span mentah LLM yang tidak ada di teks)\n\n"
        rows = []
        for b in R["main_test"]:
            h = hallucination_audit(f"test_{b}", test)
            rows.append([b, h["raw_spans"], h["hallucinated_span_rate"], h["items_with_hallucinated_span"]])
        md += tbl(["Config", "#span mentah", "Hallucinated span rate", "Item dgn span halusinasi"], rows)
        md += "\n## T14. Fairness: FPR HS pada teks non-HS yang menyebut identitas kelompok\n\n"
        rows = []
        for b in R["main_test"]:
            f = fairness(f"test_{b}", test)
            rows.append([b, f["identity"]["fpr_HS"], f["identity"]["n"], f["no_identity"]["fpr_HS"], f["no_identity"]["n"]])
        md += tbl(["Config", "FPR (identitas)", "n", "FPR (tanpa identitas)", "n"], rows)
    rt = RESULTS / "redteam.json"
    if rt.exists():
        r = json.load(open(rt, encoding="utf-8"))
        S = r["summary"]
        md += (f"\n## T15. Red-team ({S['n']} fixture): B0 lolos {S['baseline_pass']}, B6+guard v1 lolos {S.get('mitigated_v1_pass')}, "
               f"B6+guard v2 (+injection filter) lolos {S['mitigated_pass']}\n\n"
               f"False-positive injection filter pada tweet test in-scope: {S.get('injection_filter_false_positive_on_test')}. {S.get('note', '')}\n\n")
        md += tbl(["ID", "Kategori", "Surface", "Tujuan serangan", "B0", "B6+guard v1", "B6+guard v2", "Catatan (v2)"],
                  [[f["fixture_id"], f["category"], f["attack_surface"], f["goal"], "PASS" if f["baseline_B0_pass"] else "FAIL",
                    "PASS" if f.get("mitigated_v1_pass") else "FAIL", "PASS" if f["mitigated_B6_pass"] else "FAIL",
                    f["mitigation_note"][:80].replace("|", "/")] for f in r["fixtures"]])
    if "main_test" in R:
        error_analysis(test)
    open(RESULTS / "report_tables.md", "w", encoding="utf-8").write(md)
    print(md)


if __name__ == "__main__":
    main()
