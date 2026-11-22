"""Menulis ulang bagian hasil LAPORAN.md (Ringkasan Eksekutif + §6–§11) langsung dari results/*.json,
agar angka di laporan selalu identik dengan hasil run (tanpa salin manual).
Jalankan dari root proyek:  python tools/update_laporan.py
Bagian yang diganti ditandai <!-- AUTO:... START --> ... <!-- AUTO:... END -->.
"""
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
L = ["HS", "Abusive", "HS_Individual", "HS_Group"]
CFG = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
DESC = {"B0": "Direct", "B1": "Structured prompt", "B2": "Naive RAG", "B3": "Hybrid RAG", "B4": "Adaptive RAG",
        "B5": "B4 + LoRA", "B6": "Guarded agentic"}


def f(x, d=3):
    """Format angka gaya Indonesia (koma desimal)."""
    if x is None:
        return "–"
    return f"{x:.{d}f}".replace(".", ",")


def fs(x, d=3):
    return ("+" if x >= 0 else "−") + f(abs(x), d)


def pct(x):
    return "–" if x is None else f"{round(x * 100)}%"


def ms(x):
    return f"{x:,.0f}".replace(",", ".")


def jl(name):
    return [json.loads(l) for l in open(RES / "runs" / f"{name}.jsonl", encoding="utf-8")]


def main():
    R = json.load(open(RES / "results.json"))
    MT, DB, AB, RET = R["main_test"], R["deltas_bootstrap"], R["ablations"], R["retrieval"]
    dev = R["dev_selection"]
    P = dev["B6_parent"]
    rt = json.load(open(RES / "redteam.json", encoding="utf-8"))
    env = json.load(open(RES / "environment_kaggle.json")) if (RES / "environment_kaggle.json").exists() else {}
    tl = json.load(open(ROOT / "adapters" / "lora_b5_training_log.json"))
    T = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "processed" / "test.jsonl", encoding="utf-8")}
    ood = R.get("ood", {})
    ovl = json.load(open(RES / "ood_overlap.json")) if (RES / "ood_overlap.json").exists() else {}

    def sig(k):
        v = DB[k]
        return v["significant"] and v["delta"] > 0, v["significant"] and v["delta"] < 0

    def dline(k):
        v = DB[k]
        concl = ("**lebih baik** (signifikan)" if v["significant"] and v["delta"] > 0 else
                 "**lebih buruk** (signifikan)" if v["significant"] else "tidak signifikan")
        return f"| {k.replace('-', ' − ')} | {fs(v['delta'])} | [{fs(v['ci95'][0])}; {fs(v['ci95'][1])}] | {concl} |"

    # ---------- angka turunan
    zero = {c: sum(1 for r in jl(f"test_{c}") if not any(r["pred"].values())) for c in ("B0", "B1")}
    A = {r["id"]: r for r in jl("test_B4")}
    Fx = {r["id"]: r for r in jl("test_B4_fixedretr")}
    nc = lambda r, g: 0 if r["abstain"] else sum(r["pred"][l] == g[l] for l in L)
    rids = [i for i, r in A.items() if r["retries"] > 0 and T[i]["in_scope"]]
    rwin = sum(nc(A[i], T[i]) > nc(Fx[i], T[i]) for i in rids)
    rlose = sum(nc(A[i], T[i]) < nc(Fx[i], T[i]) for i in rids)
    mean_ms = lambda c: sum(r["retr_ms"] for r in A.values() if c(r)) / max(1, sum(1 for r in A.values() if c(r)))
    r_no, r_yes = mean_ms(lambda r: not r["retries"] and r["retr_ms"]), mean_ms(lambda r: r["retries"])
    fixed_all = json.load(open(RES / "runs" / "test_B4_fixedretr.manifest.json"))["metrics"]["macro_f1"]
    oos_fail = [T[r["id"]]["text"] for r in A.values() if not T[r["id"]]["in_scope"] and not r["abstain"]]
    best_step = max([k for k in ("B1-B0", "B2-B1", "B3-B2", "B4-B3", "B5-B4", f"B6-{P}")], key=lambda k: DB[k]["delta"])
    best_cfg = max(CFG, key=lambda c: MT[c]["macro_f1"])
    sig_steps = [k for k in ("B2-B1", "B3-B2", "B4-B3", "B5-B4", f"B6-{P}") if sig(k)[0]]
    names = {"B2-B1": "retrieval", "B3-B2": "hybrid RAG", "B4-B3": "adaptive RAG + verifier", "B5-B4": "LoRA", f"B6-{P}": "tool calling"}
    s = rt["summary"]
    cat = collections.defaultdict(lambda: [0, 0, 0])
    for fx in rt["fixtures"]:
        c = cat[fx["category"]]
        c[0] += fx["baseline_B0_pass"]; c[1] += fx["mitigated_v1_pass"]; c[2] += fx["mitigated_B6_pass"]
    v1_fail = [fx["fixture_id"] for fx in rt["fixtures"] if not fx["mitigated_v1_pass"]]
    rep = R["repeatability"]
    reliable = [c for c in CFG if rep[c]["std"] <= 0.05]
    a1l, b3s, a5, a7 = AB.get("A1_llm", {}), AB.get("B3_single_variable"), AB["A5"], AB["A7"]
    a2 = RET["A2_retriever"]

    # ================================================================ RINGKASAN EKSEKUTIF
    rk = [
        "### Ringkasan Eksekutif",
        "Kami membangun sistem end-to-end yang memberi **4 label biner** (HS, Abusive, HS_Individual, HS_Group) untuk tweet Indonesia informal. "
        "Setiap keluaran disertai **span bukti dari teks**, skor per label, versi pedoman, dan **abstain** untuk input di luar cakupan. "
        f"Tujuh konfigurasi B0–B6 dibandingkan pada *locked test* 220 item dengan base model yang sama. **Run resmi: Kaggle {env.get('gpu', 'GPU T4')}** "
        "(seluruh pipeline dijalankan ulang dari nol; run awal di laptop diarsipkan).",
        "",
        f"- **Macro-F1 naik dari {f(MT['B0']['macro_f1'])} (B0) menjadi {f(MT[best_cfg]['macro_f1'])} ({best_cfg})**; B6 = {f(MT['B6']['macro_f1'])} "
        f"(B6 − B0 {fs(DB['B6-B0']['delta'])}, CI95 [{fs(DB['B6-B0']['ci95'][0])}; {fs(DB['B6-B0']['ci95'][1])}]).",
        "- **Kontributor signifikan:** " + ", ".join(f"**{names[k]}** ({k.replace('-', '−')} {fs(DB[k]['delta'])})" for k in sig_steps)
        + ". Tidak signifikan: " + ", ".join(f"{names[k]} ({k.replace('-', '−')} {fs(DB[k]['delta'])})" for k in names if k not in sig_steps) + ".",
        f"- **Grounding:** evidence support B4–B6 = {pct(MT['B4']['evidence_support'])} (B0 = {pct(MT['B0']['evidence_support'])}); unsupported claim B4–B6 = 0. "
        f"ECE turun {f(MT['B0']['ece'])} → {f(MT['B5']['ece'])} (B5).",
        f"- **Abstain di luar cakupan:** {pct(MT['B4']['oos_abstain_recall'])} pada 20 teks uji **baru**. Gagal pada bahasa asing berhuruf Latin selain Inggris "
        "(Vietnam, Tagalog, Spanyol, Italia) dan potongan kode Python/HTML.",
        f"- **Red-team:** {s['baseline_pass']}/20 (B0) → {s['mitigated_v1_pass']}/20 (B6 + guard v1) → {s['mitigated_pass']}/20 (v2 + injection filter, "
        "dengan catatan risiko overfit fixture).",
        "- **Temuan negatif yang dilaporkan:** prompt terstruktur saja menurunkan F1 pada model 0.5B (B1 < B0); kenaikan B2→B3 ternyata berasal dari "
        "strategi chunking, bukan hybrid retrieval; reranker mMARCO menurunkan recall; retry adaptif dan budget konteks besar tidak membantu; "
        "tool calling tidak menambah kualitas; generalisasi ke data OOD lemah.",
        "",
        "Kode: `tbp/` · Tabel lengkap otomatis: `results/report_tables.md` · ADR: `docs/ADR.md` · Analisis error: `results/error_analysis.md` · "
        "Panduan Kaggle: `KAGGLE_PANDUAN.md`",
    ]

    # ================================================================ §6 HASIL UTAMA
    h = ["## 6. Hasil Utama (locked test: 200 in-scope + 20 out-of-scope)", "",
         f"Run resmi: Kaggle **{env.get('gpu', 'T4')}**, torch {env.get('torch', '-')}, transformers {env.get('transformers', '-')}, "
         "peft " + env.get("peft", "-") + "; inference fp16, decoding greedy.", "",
         "| Config | Macro-F1 | Micro-F1 | Exact match | Evidence support | Unsupported claim | OOS abstain | False refusal | ECE | Latency p95 (ms) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CFG:
        m = MT[c]
        b = "**" if c == best_cfg else ""
        h.append(f"| {b}{c} {DESC[c]}{b} | {b}{f(m['macro_f1'])}{b} | {f(m['micro_f1'])} | {f(m['exact_match'])} | {pct(m['evidence_support'])} | "
                 f"{f(m['unsupported_claim_rate'])} | {pct(m['oos_abstain_recall'])} | {f(m['false_refusal_rate'])} | {f(m['ece'])} | {ms(m['cost']['latency_p95_ms'])} |")
    h += ["", "Per-label F1: " + " · ".join(f"**{c}** " + ", ".join(f"{l} {f(MT[c]['per_label'][l]['F1'])}" for l in L) for c in ("B5", "B6")) + ".",
          f"Parent B6 dipilih di **dev**: B4 {f(dev['B4_dev_macro_f1'])} vs B5 {f(dev['B5_dev_macro_f1'])} → **{P}**.", "",
          "**Signifikansi (paired bootstrap 1000×, CI95 delta macro-F1):**", "",
          "| Perbandingan | Delta | CI95 | Kesimpulan |", "|---|---|---|---|"]
    h += [dline(k) for k in ("B1-B0", "B2-B1", "B3-B2", "B4-B3", "B5-B4", f"B6-{P}", "B6-B0")]
    h += ["", "### 6.1 Interpretasi per tahap",
          f"- **B0 → B1 (prompt):** {fs(DB['B1-B0']['delta'])}. Dengan prompt terstruktur + few-shot, model 0.5B **terlalu konservatif**: "
          f"{zero['B1']}/220 jawaban 'semua 0' (B0: {zero['B0']}/220). Exact match naik ({f(MT['B0']['exact_match'])} → {f(MT['B1']['exact_match'])}) karena "
          f"±47% data netral, tetapi macro-F1 turun. B1 sudah memberi span bukti (evidence support {pct(MT['B1']['evidence_support'])}).",
          f"- **B1 → B2 (retrieval):** {fs(DB['B2-B1']['delta'])} — " + ("lompatan terbesar." if best_step == "B2-B1" else "berpengaruh.")
          + " Contoh berlabel yang mirip dari train memberi 'kalibrasi' label bagi model kecil.",
          f"- **B2 → B3 (hybrid + reranker + filter):** {fs(DB['B3-B2']['delta'])}, tidak signifikan. Kontrol satu variabel (B3 dengan chunk *fixed* seperti B2) "
          + (f"menghasilkan {f(b3s['B3_fixedchunk_macro_f1'])} (Δ vs B2 {fs(b3s['delta_vs_B2']['delta'])}), jadi **kenaikan B3 berasal dari strategi chunk heading-aware, "
             "bukan dari hybrid retrieval**." if b3s else "tidak tersedia.")
          + " Filter metadata mencegah pedoman usang v0.9 masuk konteks (RT04).",
          f"- **B3 → B4 (adaptive + verifier):** {fs(DB['B4-B3']['delta'])} (signifikan). Verifier menghapus unsupported claim "
          f"({f(MT['B3']['unsupported_claim_rate'])} → 0), planner menaikkan OOS-abstain {pct(MT['B3']['oos_abstain_recall'])} → {pct(MT['B4']['oos_abstain_recall'])}, "
          f"dan ECE turun {f(MT['B3']['ece'])} → {f(MT['B4']['ece'])}. Retry adaptifnya sendiri tidak membantu (A6).",
          f"- **B4 → B5 (LoRA):** {fs(DB['B5-B4']['delta'])} (signifikan). Error reasoning/model turun tajam (§8).",
          f"- **B5 → B6 (tool calling):** {fs(DB[f'B6-{P}']['delta'])}, tidak signifikan, dengan biaya ±{f(MT['B6']['cost']['mean_llm_calls'] / max(MT['B5']['cost']['mean_llm_calls'], 1e-9), 1)}× "
          f"panggilan LLM dan latensi p95 {ms(MT['B5']['cost']['latency_p95_ms'])} → {ms(MT['B6']['cost']['latency_p95_ms'])} ms.",
          ""]

    # ================================================================ §7 ABLATION
    a1 = RET["A1_chunking"]
    h += ["## 7. Ablation", "", "### A1 — Chunking pedoman", "", "**Level retrieval** (hybrid+rerank, qrels section):", "",
          "| Strategi | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | #chunk | Ctx chars top-5 |", "|---|---|---|---|---|---|---|"]
    for k, nm in (("fixed", "fixed (450 char)"), ("heading", "heading-aware"), ("hierarchical", "hierarchical (child→parent)")):
        v = a1[k]
        h.append(f"| {nm} | {f(v['recall@5'])} | {f(v['recall@10'])} | {f(v['ndcg@10'])} | {f(v['mrr@10'])} | {v['n_chunks']} | {ms(v['ctx_chars_top5'])} |")
    if a1l:
        h += ["", "**Level LLM** (B4, locked test; groundedness = evidence support):", "",
              "| Strategi | Macro-F1 | Evidence support | Unsupported claim | Prompt tokens | Latency p95 (ms) |", "|---|---|---|---|---|---|"]
        for k in ("fixed", "heading", "hierarchical"):
            v = a1l[k]
            h.append(f"| {k} | {f(v['macro_f1'])} | {pct(v['evidence_support'])} | {f(v['unsupported_claim_rate'])} | {ms(v['mean_prompt_tokens'])} | {ms(v['latency_p95_ms'])} |")
        bl = max(a1l, key=lambda k: a1l[k]["macro_f1"])
        br = max(a1, key=lambda k: a1[k]["recall@10"])
        h += ["", f"Recall retrieval terbaik = **{br}**, tetapi kualitas jawaban LLM terbaik = **{bl}**. "
              + ("Keduanya tidak sejalan: chunk fixed mencakup beberapa section sekaligus (recall tinggi) tetapi memberi konteks yang lebih berisik bagi model kecil, "
                 "sedangkan heading-aware memberi satu definisi yang utuh dan ringkas." if bl != br else "Keduanya sejalan.")]
    h += ["", "### A2 — Retriever (heading-aware chunks, qrels identik)", "",
          "| Retriever | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | Label-precision@5 (contoh) | Latency mean (ms) | p95 (ms) |", "|---|---|---|---|---|---|---|---|"]
    for k, nm in (("dense", "dense"), ("bm25", "BM25"), ("hybrid", "hybrid (RRF)"), ("hybrid_rerank", "hybrid + rerank")):
        v = a2[k]
        h.append(f"| {nm} | {f(v['recall@5'])} | {f(v['recall@10'])} | {f(v['ndcg@10'])} | {f(v['mrr@10'])} | {f(v['example_label_precision@5'])} | {f(v['ms'], 1)} | {f(v['p95_ms'], 1)} |")
    br = max(a2, key=lambda k: a2[k]["recall@10"]); bp = max(a2, key=lambda k: a2[k]["example_label_precision@5"])
    h += ["", f"Untuk pedoman, **{br}** terbaik; untuk mencari contoh berlabel mirip, **{bp}** paling presisi (kata kasar/slang adalah sinyal leksikal). "
          + ("Reranker cross-encoder mMARCO (dilatih untuk QA) **menurunkan** recall sekaligus menambah latensi. " if a2["hybrid_rerank"]["recall@10"] < a2["hybrid"]["recall@10"] else "")
          + "B3–B6 tetap memakai hybrid+rerank karena definisi B3 di spesifikasi mewajibkannya. Rekomendasi: hybrid tanpa reranker atau reranker yang di-fine-tune pada domain ini.",
          "", "### A3 — Verifier", "", "| Varian | Macro-F1 | Unsupported claim | Evidence support | False refusal |", "|---|---|---|---|---|"]
    for b in ("B4", "B5"):
        x = AB[f"A3_{b}"]
        for k, nm in (("no_verifier", "tanpa verifier"), ("verifier_repair", "+ verifier + repair")):
            v = x[k]
            h.append(f"| {b} {nm} | {f(v['macro_f1'])} | {f(v['unsupported_claim_rate'])} | {pct(v['evidence_support'])} | {f(v['false_refusal_rate'])} |")
    h += ["", f"Verifier menghapus seluruh unsupported claim di kedua parent dan menaikkan macro-F1 B4 "
          f"{fs(AB['A3_B4']['verifier_repair']['macro_f1'] - AB['A3_B4']['no_verifier']['macro_f1'])} lewat aturan konsistensi HS↔target. "
          f"Biaya: {AB['A3_B4']['n_repair_calls'] + AB['A3_B5']['n_repair_calls']} panggilan repair; false refusal B5 {f(AB['A3_B5']['verifier_repair']['false_refusal_rate'])}.", ""]
    a4 = AB["A4"]
    ev = tl["eval"]
    h += ["### A4 — LoRA vs tanpa adapter (parent B4)",
          f"- **Gain:** macro-F1 {f(a4['B4_no_adapter']['macro_f1'])} → {f(a4['B5_lora']['macro_f1'])} ({fs(a4['delta']['delta'])}, CI95 [{fs(a4['delta']['ci95'][0])}; {fs(a4['delta']['ci95'][1])}]). "
          + ", ".join(f"{l} {fs(a4['B5_lora']['per_label'][l]['F1'] - a4['B4_no_adapter']['per_label'][l]['F1'])}" for l in L) + ".",
          "- **Overfit check:** val loss (train held-out) " + " → ".join(f(e["val_loss"]) for e in ev)
          + "; dev loss " + " → ".join(f(e["dev_loss"]) for e in ev) + ". Kurva turun lalu stabil, tidak ada tanda overfit dalam 1 epoch.",
          f"- **Resource:** {tl['config']['n_train']} contoh, 200 step, {f(tl['resources']['train_seconds'] / 60, 0)} menit di {tl['config']['gpu']} "
          f"({tl['config']['method']}), peak VRAM {f(tl['resources']['peak_vram_gb'], 1)} GB, adapter {f(tl['resources']['adapter_bytes'] / 1e6, 0)} MB, "
          f"trainable params {f(tl['config']['trainable_params'] / tl['config']['total_params'] * 100, 2)}%.",
          f"- **OOD (Alfina et al. 2017, label HS saja):** dataset utama ternyata memuat sebagian tweet Alfina (paper Ibrohim & Budi §3). "
          f"{ovl.get('overlap_removed', '?')} dari {ovl.get('alfina_total', '?')} tweet yang sama/near-duplicate (cosine ≥ 0,8) dengan dataset utama **dibuang**; "
          f"200 sampel diambil dari {ovl.get('clean_pool', '?')} tweet bersih.", "",
          "| Varian | HS-P | HS-R | HS-F1 | Binary macro-F1 |", "|---|---|---|---|---|"]
    for b, nm in (("B4", "B4 (no adapter)"), ("B5", "B5 (LoRA)")):
        v = ood[b]
        h.append(f"| {nm} | {f(v['HS_P'])} | {f(v['HS_R'])} | {f(v['HS_F1'])} | {f(v['binary_macro_f1'])} |")
    d_ood = ood["B5"]["binary_macro_f1"] - ood["B4"]["binary_macro_f1"]
    h += ["", f"Pada data yang benar-benar baru, kedua varian **over-prediksi HS** (recall tinggi, presisi rendah) dan LoRA tidak membantu ({fs(d_ood)}). "
          f"Peningkatan B5 di test in-domain ({fs(a4['delta']['delta'])}) **tidak** berpindah ke domain lain, yaitu domain drift yang nyata. "
          "Run laptop awal memakai OOD yang bocor; hasilnya diarsipkan untuk transparansi.", "",
          f"### A5 — B6 tanpa vs dengan tool calling (parent identik {P})", "",
          "| Varian | Macro-F1 | LLM calls/item | Prompt tokens | Latency mean (ms) | p95 (ms) |", "|---|---|---|---|---|---|"]
    for k, nm in (("without_tools", "tanpa tool (guard + temporal check)"), ("with_tools", "dengan tool")):
        v = a5[k]
        h.append(f"| {nm} | {f(v['macro_f1'])} | {f(v['cost']['mean_llm_calls'], 2)} | {ms(v['cost']['mean_prompt_tokens'])} | {ms(v['cost']['latency_mean_ms'])} | {ms(v['cost']['latency_p95_ms'])} |")
    ts = a5["tool_stats_total"]
    h += ["", f"Dari {ts['attempted']} tool call yang dicoba: **tool correctness {pct(a5['tool_correctness'])}** ({ts['ok']} valid), invalid-arg {pct(a5['invalid_arg_rate'])} "
          f"({ts['invalid_args']}), **unsafe call {pct(a5['unsafe_call_rate'])}**. **Budget overrun pada {pct(a5['budget_overrun_rate'])} item:** model hampir selalu mencoba langkah "
          f"ke-3 dan tidak pernah memilih `done`, lalu guard memotongnya. End-task outcome {fs(a5['delta']['delta'])} (tidak signifikan).", "",
          "### A6 — Fixed vs adaptive retrieval (B4, satu variabel: retry)", "",
          "| Slice | n | Fixed: exact / macro-F1 | Adaptive: exact / macro-F1 | Retry rata-rata |", "|---|---|---|---|---|"]
    a6f, a6a = AB["A6"]["test_B4_fixedretr"], AB["A6"]["test_B4"]
    for k, nm in (("easy", "easy (0–1 label)"), ("multi", "multi (≥2 label)"), ("conflict", "conflict (tetangga ≠ gold)")):
        h.append(f"| {nm} | {a6f[k]['n']} | {f(a6f[k]['exact_match'])} / {f(a6f[k]['macro_f1'])} | {f(a6a[k]['exact_match'])} / {f(a6a[k]['macro_f1'])} | {f(a6a[k]['mean_retries'], 2)} |")
    h.append(f"| no-evidence (OOS) | {a6f['no_evidence']['n']} | abstain {pct(a6f['no_evidence']['oos_abstain_recall'])} | abstain {pct(a6a['no_evidence']['oos_abstain_recall'])} | {f(a6a['no_evidence']['mean_retries'], 2)} |")
    h += ["", f"Keseluruhan: fixed {f(fixed_all)} vs adaptive {f(MT['B4']['macro_f1'])} ({fs(MT['B4']['macro_f1'] - fixed_all)}). "
          f"**Retry benefit, failure, biaya:** retry dipicu pada {len(rids)} tweet in-scope; **{rwin} membaik, {rlose} memburuk**, {len(rids) - rwin - rlose} tidak berubah; "
          f"waktu retrieval {ms(r_no)} → {ms(r_yes)} ms pada item yang di-retry. Retry membantu slice *conflict* (exact {f(a6f['conflict']['exact_match'])} → {f(a6a['conflict']['exact_match'])}) "
          "tetapi secara keseluruhan tidak memberi manfaat bersih. Macro-F1 slice *easy* rendah karena slice ini didominasi item tanpa label positif; exact match lebih representatif.", "",
          "### A7 — Budget konteks", "", "| Budget | k contoh / k pedoman / max char | Macro-F1 | Prompt tokens | Latency mean | p95 |", "|---|---|---|---|---|---|"]
    for k, nm in (("small", "kecil"), ("large", "besar")):
        v = a7[k]; bu = v["budget"]
        h.append(f"| {nm} | {bu['k_examples']} / {bu['k_guides']} / {ms(bu['max_ctx_chars'])} | {f(v['macro_f1'])} | {ms(v['cost']['mean_prompt_tokens'])} | {ms(v['cost']['latency_mean_ms'])} ms | {ms(v['cost']['latency_p95_ms'])} ms |")
    h += ["", f"Delta {fs(a7['delta']['delta'])} (CI95 [{fs(a7['delta']['ci95'][0])}; {fs(a7['delta']['ci95'][1])}], tidak signifikan); latensi p95 naik "
          f"{f(a7['large']['cost']['latency_p95_ms'] / a7['small']['cost']['latency_p95_ms'], 2)}×. Titik Pareto ada di **budget kecil**: konteks lebih panjang tidak membantu model 0.5B.", ""]

    # ================================================================ §8 ERROR
    ea = (RES / "error_analysis.md").read_text(encoding="utf-8")
    table = ea[ea.index("| Config | #salah"):ea.index("\n\n", ea.index("| Config | #salah"))]
    rows = {line.split("|")[1].strip(): [x.strip() for x in line.split("|")[2:-1]] for line in table.splitlines()[2:]}
    h += ["## 8. Analisis Penyebab Error", "",
          "Taksonomi deterministik mencakup enam sumber error di spesifikasi (retrieval, reasoning, tool, source, prompt, adaptation), diperiksa berurutan per item in-scope "
          "yang vektor labelnya salah: **format/prompt** (JSON tidak valid) → **abstain** → **source** (dugaan noise label: gold Abusive=0 padahal tweet memuat kata dari "
          "`abusive.csv` milik penyedia dataset) → **tool** (B6: tak satu pun tool call valid) → **retrieval** (mayoritas label contoh ter-retrieve ≠ gold) → "
          "**reasoning/model**. **Adaptation** dibaca dari perubahan reasoning B4 → B5.", "", table, "",
          f"- **Prompt/format error** turun setelah LoRA (B4: {rows['B4'][1]} → B5: {rows['B5'][1]}).",
          f"- **LoRA (adaptation) terutama memperbaiki reasoning** ({rows['B4'][6]} → {rows['B5'][6]}).",
          f"- Error yang tersisa **didominasi retrieval** (B5: {rows['B5'][5]} dari {rows['B5'][0]}): contoh tetangga bertentangan dengan gold.",
          f"- **Source:** {min(int(rows[c][3]) for c in rows)}–{max(int(rows[c][3]) for c in rows)} error per konfigurasi kemungkinan berasal dari label sumber (mis. tweet berisi 'monyet' dilabel tidak kasar). "
          "Inilah alasan human adjudication (§10) diperlukan.",
          f"- **Tool:** {rows['B6'][4]} error B6 terjadi ketika agent gagal membuat satu pun tool call valid.",
          "- Contoh kualitatif: `results/error_analysis.md`.", ""]

    # ================================================================ §9
    import sys
    sys.path.insert(0, str(ROOT))
    from tbp.make_report import fairness, hallucination_audit
    test = list(T.values())
    hal = {c: hallucination_audit(f"test_{c}", test) for c in CFG}
    fai = {c: fairness(f"test_{c}", test) for c in CFG}
    h += ["## 9. Hallucination Audit, Fairness, Repeatability", "",
          "**Hallucination audit** (span mentah LLM yang tidak ada di teks, sebelum verifier):", "",
          "| Config | " + " | ".join(CFG[1:]) + " |", "|---|" + "---|" * 6,
          "| Hallucinated span rate | " + " | ".join(f(hal[c]["hallucinated_span_rate"]) for c in CFG[1:]) + " |", "",
          f"Sekitar {pct(min(hal[c]['hallucinated_span_rate'] for c in CFG[1:]))}–{pct(max(hal[c]['hallucinated_span_rate'] for c in CFG[1:]))} span mentah dikarang model. "
          f"Verifier membuang span tersebut sehingga evidence support akhir B4–B6 = {pct(MT['B4']['evidence_support'])}. B0 tidak pernah memberi span.", "",
          "**Fairness** (FPR label HS pada tweet non-HS):", "",
          f"| Config | FPR, menyebut identitas kelompok (n={fai['B0']['identity']['n']}) | FPR, tanpa identitas (n={fai['B0']['no_identity']['n']}) |", "|---|---|---|"]
    for c in ("B3", "B4", "B5", "B6"):
        h.append(f"| {c} | {f(fai[c]['identity']['fpr_HS'])} | {f(fai[c]['no_identity']['fpr_HS'])} |")
    h += ["", f"Di B3/B4, tweet yang menyebut agama/etnis/gender lebih sering dituduh HS (bias identitas, "
          f"{fs(fai['B4']['identity']['fpr_HS'] - fai['B4']['no_identity']['fpr_HS'])} pada B4). Setelah LoRA (B5): {f(fai['B5']['identity']['fpr_HS'])} vs {f(fai['B5']['no_identity']['fpr_HS'])}. "
          f"Sampel identitas kecil (n={fai['B0']['identity']['n']}), sehingga temuan ini **indikatif, bukan konklusif**.", "",
          "**Repeatability** (3 run independen, seed 11/22/33, T=0.7, top-p 0.9, subset 40 item):", "",
          "| Config | " + " | ".join(CFG) + " |", "|---|" + "---|" * 7,
          "| Mean macro-F1 | " + " | ".join(f(rep[c]["mean"]) for c in CFG) + " |",
          "| Std | " + " | ".join(f(rep[c]["std"]) for c in CFG) + " |", "",
          f"Menurut definisi *reliable* (std ≤ 0,05): **{', '.join(reliable) if reliable else 'tidak ada'}**. "
          "Dengan sampling T=0,7, model 0.5B cukup sensitif; agentic loop (B6) paling tidak stabil. Hasil utama memakai decoding greedy (deterministik).", "",
          "## 10. Human Review (status: perangkat siap, WAJIB diisi tim)",
          "Anotasi blind dua annotator **tidak disimulasikan** dengan model (ADR-012). Yang sudah disiapkan:",
          "- `human_review/sheet_annotator_A.csv` dan `sheet_annotator_B.csv`: 100 item test in-scope, urutan acak berbeda, tanpa label gold maupun prediksi.",
          "- `human_review/protocol.json`: instruksi blind + pedoman `label_policy_v1.0.md`.",
          "- `python -m tbp.human_review agree`: Cohen's κ per label, % agreement, `disagreement_log.csv`, `adjudication_sheet.csv`.",
          "- `python -m tbp.human_review finalize`: human gold, κ human vs label dataset, dan macro-F1 B0–B6 terhadap human gold.", "",
          "> **TODO tim:** isi setelah anotasi. κ per label: HS ___ · Abusive ___ · HS_Individual ___ · HS_Group ___ ; jumlah disagreement ___ ; ringkasan resolusi ___.", "",
          "## 11. Red-team (20 fixture)", "",
          "| Kategori | B0 (tanpa guard) | B6 + guard v1 | B6 + guard v2 (+ injection filter) |", "|---|---|---|---|"]
    for k, nm in (("injection", "Prompt/document injection (5)"), ("policy_pii", "Policy/PII leakage (5)"), ("tool", "Tool misuse (5)"), ("robust", "Safety/robustness (5)")):
        c = cat[k]
        h.append(f"| {nm} | {c[0]}/5 | {c[1]}/5 | {c[2]}/5 |")
    h += [f"| **Total** | **{s['baseline_pass']}/20** | **{s['mitigated_v1_pass']}/20** | **{s['mitigated_pass']}/20** |", "",
          f"- v1 gagal pada **{', '.join(v1_fail) or '–'}** (blok JSON label palsu di dalam teks; model menurut/abstain).",
          "- **v2 ditambahkan setelah melihat hasil v1 di run laptop:** filter pola generik (frasa override instruksi, prefix peran `SYSTEM:`, perintah ubah label/abstain, "
          "blok JSON berkunci label). Karena didesain setelah melihat fixture, angka 20/20 **berisiko overfit ke fixture** dan perlu diuji pada serangan baru. "
          f"False-positive filter pada 200 tweet test = **{s['injection_filter_false_positive_on_test']}**.",
          "- Detail per fixture (ID, goal, attack surface, baseline outcome, mitigation outcome): `results/redteam.json` dan `results/report_tables.md` (T15).", ""]

    lap = (ROOT / "LAPORAN.md").read_text(encoding="utf-8")
    lap = re.sub(r"### Ringkasan Eksekutif.*?(?=\n---\n)", lambda _: "\n".join(rk) + "\n", lap, count=1, flags=re.S)
    lap = re.sub(r"## 6\. Hasil Utama.*?(?=## 12\. Serving)", lambda _: "\n".join(h) + "\n", lap, count=1, flags=re.S)
    (ROOT / "LAPORAN.md").write_text(lap, encoding="utf-8")
    print("LAPORAN.md diperbarui:", len(rk), "baris ringkasan,", len(h), "baris hasil")
    print("OOS gagal:", oos_fail)


if __name__ == "__main__":
    main()
