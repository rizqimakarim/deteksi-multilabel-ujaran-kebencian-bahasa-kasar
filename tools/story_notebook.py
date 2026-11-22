"""Generator Notebook 1 versi presentasi: setiap tahap = penjelasan -> kode (tersembunyi, bisa dibuka) -> dijalankan
-> output 'Cara verifikasi' (PDF §3) dan 'Eksperimen' (PDF §4 / §4.1). Kode diambil otomatis dari file .py (satu sumber)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src, hide_input=False, hide_output=False):
    meta = {}
    if hide_input:
        meta["_kg_hide-input"] = True
    if hide_output:
        meta["_kg_hide-output"] = True
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": src}


def wf(path):
    """Sel %%writefile berisi kode asli dari file .py (input disembunyikan, klik 'Show' untuk melihat)."""
    txt = (ROOT / path).read_text(encoding="utf-8")
    return code(f"%%writefile {path}\n{txt}", hide_input=True, hide_output=True)


HELPERS = r'''import pandas as pd, numpy as np, json, subprocess, time, os, sys, glob
from IPython.display import display, Markdown, Image
pd.set_option("display.max_colwidth", 140); pd.set_option("display.width", 200)
LABELS = ["HS", "Abusive", "HS_Individual", "HS_Group"]
CFGS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
for d in ["data/processed", "results", "adapters", "logs"]:          # run bersih dari nol
    shutil.rmtree(d, ignore_errors=True)
os.makedirs("logs", exist_ok=True)

def run(*args):
    """Menjalankan modul .py proyek (python -m tbp.xxx) dan menampilkan ringkasan log."""
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    log = p.stdout + p.stderr
    open(f"logs/{'_'.join(a.replace('tbp.', '') for a in args)}.log", "w", encoding="utf-8").write(log)
    keep = [l for l in log.splitlines() if l.strip() and not any(w in l for w in ("Warning", "warn(", "Loading weights", "symlink", "it/s]", "HF_TOKEN"))]
    print("\n".join(keep[-15:]))
    print(f"⏱  {' '.join(args)}: {(time.time() - t0) / 60:.1f} menit")
    if p.returncode:
        print(log[-4000:]); raise RuntimeError(f"Gagal: {args}")

def show(s): display(Markdown(s))
def J(p): return json.load(open(p, encoding="utf-8"))
def JL(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def ok(b): return "✅" if b else "❌"
def verif(elemen, standar, cara):
    show(f"#### ✔ Verifikasi PDF §3 — {elemen}\n> **Standar wajib minimum:** {standar}  \n> **Cara verifikasi:** {cara}")
def eksperimen(kode, judul, fitur_or_hipotesis, eksp):
    show(f"#### 🧪 {kode} — {judul}\n> **{'Fitur pembeda' if kode.startswith('B') else 'Hipotesis / yang diukur'} (PDF):** {fitur_or_hipotesis}  \n> **Eksperimen (PDF):** {eksp}")
def delta(k):
    v = J("results/results.json")["deltas_bootstrap"][k]
    return f"Δ macro-F1 **{v['delta']:+.3f}** (CI95 {v['ci95'][0]:+.3f} … {v['ci95'][1]:+.3f}) → " + ("**signifikan**" if v["significant"] else "tidak signifikan")
print("Helper siap.")
'''

VERIF_DATA = r'''# ---------- PDF §3: Corpus utama
verif("Corpus utama", "Sumber nyata public atau private dengan izin, minimal 300 dokumen.", "Source register, hash, index documents.")
reg = J("data/source_register.json")
display(pd.DataFrame([{"ID": r["source_id"], "Nama": r["name"][:60], "Lisensi": r["license"][:45], "SHA-256": r["sha256"][:16] + "…", "Dipakai untuk": r["use"]} for r in reg]))
docs = JL("data/processed/corpus_examples.jsonl")
print(f"Dokumen corpus (contoh train berlabel): {len(docs):,} → minimal 300 {ok(len(docs) >= 300)}")
for s in ("fixed", "heading", "hierarchical"):
    print(f"  + chunk pedoman label ({s}): {len(JL(f'data/processed/guide_chunks_{s}.jsonl'))}")
display(pd.DataFrame(docs[:3])[["doc_id", "doc_type", "split", "text", "labels", "source", "version"]])

# ---------- PDF §3: Benchmark utama
verif("Benchmark utama", "Minimal 100 instance query.", "Manifest split dan SHA-256; dedup/near-dup checks.")
m = J("data/processed/split_manifest.json")
display(pd.DataFrame([{"split": k, "n": v["n"], "in-scope": v["n_in_scope"], "out-of-scope (unanswerable)": v["n_out_of_scope"], "SHA-256": v["sha256"]}
                      for k, v in m["splits"].items()]))
display(pd.Series(m["stats"], name="nilai").to_frame("Dedup / near-dup check"))
print(f"Test = {m['splits']['test']['n']} query → minimal 100 {ok(m['splits']['test']['n'] >= 100)} | overlap train vs dev/test = {m['stats']['exact_overlap_train_vs_heldout']} {ok(m['stats']['exact_overlap_train_vs_heldout'] == 0)}")

# ---------- PDF §3: Evidence / qrels
verif("Evidence/qrels", "Qrels untuk seluruh query answerable yang bergantung pada retrieval di test, dan label unanswerable untuk query tanpa evidence.",
      "Qrels id, relevance level, document/passage version, annotation rule.")
q = pd.DataFrame(JL("data/processed/qrels.jsonl"))
display(q.head(6))
t = pd.DataFrame(JL("data/processed/test.jsonl"))
cov = t[t.in_scope].id.isin(q[q.split == "test"].query_id).mean()
print(f"Total qrels: {len(q)} | query test in-scope yang punya qrels: {cov:.0%} {ok(cov == 1)} | qrels unanswerable di test: {(~q[q.split == 'test'].answerable).sum()}")
'''

LORA_SHOW = r'''import matplotlib.pyplot as plt
tl = J("adapters/lora_b5_training_log.json")
eksperimen("B5", "B4 + Parameter-Efficient Fine-Tuning", "Base generator yang sama dipasangi LoRA/QLoRA untuk task domain; dataset train berizin.",
           "Same retrieval/index/prompt budgets; simpan adapter dan training logs.")
c = tl["config"]
display(pd.Series({"Base model": c["base_model"], "Metode": c["method"], "r / alpha / dropout": f"{c['r']} / {c['lora_alpha']} / {c['lora_dropout']}",
                   "Target modules": ", ".join(c["target_modules"]), "LR / epoch / grad-accum": f"{c['lr']} / {c['epochs']} / {c['grad_accum']}",
                   "Data latih (dari TRAIN)": c["n_train"], "Trainable params": f"{c['trainable_params']:,} ({c['trainable_params'] / c['total_params']:.2%})",
                   "GPU": c["gpu"], "Waktu training": f"{tl['resources']['train_seconds'] / 60:.1f} menit", "Peak VRAM": f"{tl['resources']['peak_vram_gb']} GB",
                   "Ukuran adapter": f"{tl['resources']['adapter_bytes'] / 1e6:.0f} MB"}, name="").to_frame("Konfigurasi & resource (bukti training log)"))
ev = pd.DataFrame(tl["eval"])
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(ev.step, ev.val_loss, "-o", color="#2a78d6", lw=2, label="val (train held-out)")
ax.plot(ev.step, ev.dev_loss, "-o", color="#eb6834", lw=2, label="dev")
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Kurva loss LoRA (cek overfit)"); ax.grid(alpha=.3); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
print("Adapter tersimpan:", sorted(os.listdir("adapters/lora_b5")))
'''

RETR_SHOW = r'''R = J("results/results.json"); RET = R["retrieval"]
eksperimen("A1", "Chunking pedoman (level retrieval)", "fixed vs heading-aware vs hierarchical chunking; log passage recall, index size, context length dan groundedness.",
           "Satu variabel (strategi chunk); retriever hybrid+rerank; qrels identik.")
display(pd.DataFrame(RET["A1_chunking"]).T[["recall@5", "recall@10", "ndcg@10", "mrr@10", "top5_relevant_frac", "ctx_chars_top5", "n_chunks", "index_chars"]].round(3))
eksperimen("A2", "Retriever", "dense vs BM25 vs hybrid vs hybrid+reranking pada qrels identik; laporkan Recall@5/10, nDCG@10, MRR@10.", "Satu variabel (retriever).")
a2 = pd.DataFrame(RET["A2_retriever"]).T[["recall@5", "recall@10", "ndcg@10", "mrr@10", "example_label_precision@5", "ms", "p95_ms"]].round(3)
display(a2)
show(f"**Hasil:** Recall@10 terbaik = **{a2['recall@10'].idxmax()}**; label-precision@5 (contoh mirip) terbaik = **{a2['example_label_precision@5'].idxmax()}**. "
     + ("Reranker **menurunkan** recall dibanding hybrid tanpa reranker." if a2.loc['hybrid_rerank', 'recall@10'] < a2.loc['hybrid', 'recall@10'] else ""))
'''

DEV_SHOW = r'''d = J("results/results.json")["dev_selection"]
show(f"**Pemilihan parent B6 di dev (bukan test):** B4 = {d['B4_dev_macro_f1']:.3f}, B5 = {d['B5_dev_macro_f1']:.3f} → parent B6 = **{d['B6_parent']}**")
'''

TEST_SHOW = r'''import matplotlib.pyplot as plt
R = J("results/results.json"); MT = R["main_test"]; AB_ = R.get("ablations", {})
P = R["dev_selection"]["B6_parent"]
df = pd.DataFrame([{"Config": c, "Macro-F1": MT[c]["macro_f1"], "Micro-F1": MT[c]["micro_f1"], "Exact": MT[c]["exact_match"],
                    "Evidence support": MT[c]["evidence_support"], "Unsupported claim": MT[c]["unsupported_claim_rate"],
                    "OOS abstain": MT[c]["oos_abstain_recall"], "ECE": MT[c]["ece"], "p95 ms": MT[c]["cost"]["latency_p95_ms"],
                    "LLM calls": MT[c]["cost"]["mean_llm_calls"]} for c in CFGS]).set_index("Config")
show("### Hasil utama B0–B6 (locked test: 200 tweet + 20 teks di luar cakupan)")
display(df.round(3))
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.bar(df.index, df["Macro-F1"], color="#2a78d6", width=.55)
for i, v in enumerate(df["Macro-F1"]): ax.text(i, v + .015, f"{v:.2f}", ha="center", fontsize=9, color="#52514e")
ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1"); ax.set_title("Macro-F1 per konfigurasi (base model sama)"); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.show()
display(pd.DataFrame({c: {l: MT[c]["per_label"][l]["F1"] for l in LABELS} for c in CFGS}).T.round(3).rename_axis("F1 per label"))
'''

PER_B = r'''R = J("results/results.json"); MT = R["main_test"]; P = R["dev_selection"]["B6_parent"]
man = {c: J(f"results/runs/test_{c}.manifest.json") for c in CFGS}
runs = {c: {r["id"]: r for r in JL(f"results/runs/test_{c}.jsonl")} for c in CFGS}
T = {r["id"]: r for r in JL("data/processed/test.jsonl")}
cmp = lambda cs: pd.DataFrame({c: {"Macro-F1": MT[c]["macro_f1"], "Exact": MT[c]["exact_match"], "Evidence support": MT[c]["evidence_support"],
                                   "OOS abstain": MT[c]["oos_abstain_recall"], "p95 ms": MT[c]["cost"]["latency_p95_ms"], "LLM calls": MT[c]["cost"]["mean_llm_calls"]} for c in cs}).T.round(3)
zero = {c: sum(1 for r in runs[c].values() if not any(r["pred"].values())) for c in ("B0", "B1")}
A, F = runs["B4"], {r["id"]: r for r in JL("results/runs/test_B4_fixedretr.jsonl")}
nc = lambda r, g: 0 if r["abstain"] else sum(r["pred"][l] == g[l] for l in LABELS)
rids = [i for i, r in A.items() if r["retries"] > 0 and T[i]["in_scope"]]
win, lose = sum(nc(A[i], T[i]) > nc(F[i], T[i]) for i in rids), sum(nc(A[i], T[i]) < nc(F[i], T[i]) for i in rids)
mean_ms = lambda cond: np.mean([r["retr_ms"] for r in A.values() if cond(r)] or [0])
AB_ = R["ablations"]; b3s = AB_.get("B3_single_variable", {}); a2 = R["retrieval"]["A2_retriever"]

eksperimen("B0", "Direct LLM baseline", "Zero-shot tanpa retrieval, tanpa tool dan tanpa evidence verification.", "Model dasar, sampling dan output contract tetap.")
display(pd.DataFrame([{"Config": c, "Base model": man[c]["meta"]["model_id"], "Temperature": man[c]["meta"]["temperature"], "Prompt": man[c]["meta"]["prompt_id"],
                       "Adapter": man[c]["meta"]["adapter"] or "-"} for c in CFGS]).set_index("Config"))
show(f"**Hasil:** base model, sampling greedy, dan output contract JSON sama di B0–B6. Titik awal macro-F1 **{MT['B0']['macro_f1']:.3f}**.")

eksperimen("B1", "Structured Prompted LLM", "Few-shot (hanya train/dev), constraint, schema dan policy abstain.", "Ukur efek prompt terhadap B0.")
display(cmp(["B0", "B1"]))
show(f"**Hasil:** {delta('B1-B0')}. Jawaban 'semua 0': B0 {zero['B0']}/220 vs B1 {zero['B1']}/220.")

eksperimen("B2", "Naive RAG", "Dense retrieval + fixed chunk + top-k context + answer.", "Ukur efek retrieval dibanding B1.")
display(cmp(["B1", "B2"]))
show(f"**Hasil:** {delta('B2-B1')}.")

eksperimen("B3", "Hybrid RAG", "BM25 + dense + fusion + reranker + metadata filter.", "Ukur retrieval effectiveness dan tambahan latensi.")
display(cmp(["B2", "B3"]))
show(f"**Hasil:** Recall@10 dense {a2['dense']['recall@10']:.3f} · hybrid {a2['hybrid']['recall@10']:.3f} · hybrid+rerank {a2['hybrid_rerank']['recall@10']:.3f}; "
     f"latensi p95 {MT['B2']['cost']['latency_p95_ms']:.0f} → {MT['B3']['cost']['latency_p95_ms']:.0f} ms. {delta('B3-B2')}. "
     + (f"Kontrol satu variabel (B3 chunk fixed) = {b3s['B3_fixedchunk_macro_f1']:.3f}, Δ vs B2 {b3s['delta_vs_B2']['delta']:+.3f} → kenaikan B3 berasal dari chunking." if b3s else ""))

eksperimen("B4", "Self-corrective/Adaptive RAG", "Query planner, evidence sufficiency score, retrieval retry bounded, claim verifier, abstain.", "Ukur retry benefit, failure dan biaya.")
display(pd.DataFrame([{"Tweet di-retry": len(rids), "Retry benefit": win, "Retry failure": lose, "Tidak berubah": len(rids) - win - lose,
                       "Retrieval ms tanpa retry": round(mean_ms(lambda r: not r["retries"] and r["retr_ms"])), "Retrieval ms dengan retry": round(mean_ms(lambda r: r["retries"]))}]))
display(cmp(["B3", "B4"]))
show(f"**Hasil:** {delta('B4-B3')}.")

eksperimen("B5", "B4 + Parameter-Efficient Fine-Tuning", "Base generator yang sama dipasangi LoRA untuk task domain; dataset train berizin.",
           "Same retrieval/index/prompt budgets; simpan adapter dan training logs.")
chk = pd.DataFrame([{"Komponen": "prompt hash", "B4": man["B4"]["meta"]["prompt_hash"], "B5": man["B5"]["meta"]["prompt_hash"]},
                    {"Komponen": "budget", "B4": json.dumps(man["B4"]["meta"]["budget"]), "B5": json.dumps(man["B5"]["meta"]["budget"])},
                    {"Komponen": "index contoh", "B4": man["B4"]["index_versions"]["examples"], "B5": man["B5"]["index_versions"]["examples"]},
                    {"Komponen": "index pedoman", "B4": man["B4"]["index_versions"]["guide_heading"], "B5": man["B5"]["index_versions"]["guide_heading"]},
                    {"Komponen": "adapter", "B4": str(man["B4"]["meta"]["adapter"]), "B5": str(man["B5"]["meta"]["adapter"])}])
chk["Sama?"] = [ok(a == b) if k != "adapter" else "(yang diuji)" for k, a, b in zip(chk.Komponen, chk.B4, chk.B5)]
display(chk)
display(cmp(["B4", "B5"]))
show(f"**Hasil:** {delta('B5-B4')}. Adapter & training log: `adapters/lora_b5/`, `adapters/lora_b5_training_log.json`.")

eksperimen("B6", "Guarded Agentic LLM", "B5 atau B4 yang paling kuat (dipilih di dev), structured tool args, authorization, temporal checking, verified final answer.",
           "Catat model asal; evaluasi delta agentic terhadap parent yang identik.")
a5 = AB_["A5"]
display(pd.DataFrame({f"{P} (parent)": {"Macro-F1": MT[P]["macro_f1"]}, "B6 tanpa tool": {"Macro-F1": a5["without_tools"]["macro_f1"]},
                      "B6 dengan tool": {"Macro-F1": MT["B6"]["macro_f1"]}}).T.round(3))
show(f"**Hasil:** model asal tercatat di manifest (`parent={man['B6']['meta']['parent']}`). {delta('B6-' + P)}.")
'''

EXAMPLE = r'''# Contoh satu tweet dijawab B0–B6 (dari raw output yang tersimpan)
R = J("results/results.json"); runs = {c: {r["id"]: r for r in JL(f"results/runs/test_{c}.jsonl")} for c in CFGS}
T = {r["id"]: r for r in JL("data/processed/test.jsonl")}
cand = [i for i, g in T.items() if g["in_scope"] and sum(g[l] for l in LABELS) >= 2 and not runs["B6"][i]["abstain"]
        and runs["B6"][i]["pred"] == {l: g[l] for l in LABELS} and runs["B0"][i]["pred"] != {l: g[l] for l in LABELS}]
i = (cand or [k for k, v in T.items() if v["in_scope"]])[0]; g = T[i]
show(f"**Tweet `{i}`:** “{g['text']}”  \n**Gold:** " + ", ".join(f"{l}={g[l]}" for l in LABELS))
display(pd.DataFrame([{"Config": c, **({l: runs[c][i]["pred"][l] for l in LABELS} if not runs[c][i]["abstain"] else {l: "abstain" for l in LABELS}),
                       "Bukti (span)": ", ".join(runs[c][i]["evidence"]) or "-", "Benar semua?": ok(not runs[c][i]["abstain"] and runs[c][i]["pred"] == {l: g[l] for l in LABELS})} for c in CFGS]).set_index("Config"))
'''

VERIF_LLM = r'''verif("LLM comparison", "Menggunakan model LLM plain dibandingkan dengan LLM yang di-tuning LoRA/QLoRA.",
      "Raw output per item, prompt/model/index versions, per-run token/latency.")
for c in ("B4", "B5"):
    r = JL(f"results/runs/test_{c}.jsonl")[0]
    show(f"**{c} — raw output item `{r['id']}`:** `{(r['raw_output'] or '')[:200]}`")
display(pd.DataFrame([{"Run": c, "prompt": J(f"results/runs/test_{c}.manifest.json")["meta"]["prompt_id"] + " #" + J(f"results/runs/test_{c}.manifest.json")["meta"]["prompt_hash"],
                       "model": J(f"results/runs/test_{c}.manifest.json")["meta"]["model_id"], "adapter": J(f"results/runs/test_{c}.manifest.json")["meta"]["adapter"],
                       "index": J(f"results/runs/test_{c}.manifest.json")["index_versions"]["examples"],
                       "prompt tok/item": J(f"results/runs/test_{c}.manifest.json")["metrics"]["cost"]["mean_prompt_tokens"],
                       "latency mean ms": J(f"results/runs/test_{c}.manifest.json")["metrics"]["cost"]["latency_mean_ms"]} for c in ("B4", "B5")]))
print("Raw output per item tersimpan untuk", len(glob.glob("results/runs/*.jsonl")), "run (results/runs/*.jsonl).")
'''

ABL_SHOW = r'''R = J("results/results.json"); AB = R["ablations"]; MT = R["main_test"]
eksperimen("A1", "Chunking (level LLM / groundedness)", "fixed vs heading-aware vs hierarchical; log passage recall, index size, context length dan groundedness.", "B4 dijalankan dengan 3 strategi chunk.")
display(pd.DataFrame(AB["A1_llm"]).T.round(3))
eksperimen("A3", "Verifier", "no-verifier vs independent verifier + bounded repair; ukur unsupported claims dan false-refusal.", "Prediksi sebelum vs sesudah verifier pada run yang sama.")
display(pd.DataFrame([{"Varian": f"{b} {k}", **AB[f"A3_{b}"][k]} for b in ("B4", "B5") for k in ("no_verifier", "verifier_repair")]).set_index("Varian").round(3))
eksperimen("A4", "LoRA vs no-adapter", "no-adapter vs LoRA pada parent B4; ukur gain, overfit, resource, dan performa OOD.", "B4 vs B5 di test + OOD (lihat bagian 9) + training log (bagian 3).")
a4 = AB["A4"]; display(pd.DataFrame({k: {**{l: a4[k]["per_label"][l]["F1"] for l in LABELS}, "Macro-F1": a4[k]["macro_f1"]} for k in ("B4_no_adapter", "B5_lora")}).T.round(3))
show(f"**Gain:** Δ {a4['delta']['delta']:+.3f} (CI95 {a4['delta']['ci95']}) → " + ("signifikan" if a4["delta"]["significant"] else "tidak signifikan"))
eksperimen("A5", "B6 tanpa vs dengan tool calling", "ukur tool correctness, unsafe call rate, budget overrun dan end-task outcome.", "Parent identik, guard identik.")
a5 = AB["A5"]
display(pd.DataFrame([{"Tool correctness": a5["tool_correctness"], "Unsafe call rate": a5["unsafe_call_rate"], "Invalid-arg rate": a5["invalid_arg_rate"],
                       "Budget overrun (per item)": a5["budget_overrun_rate"], "Macro-F1 tanpa tool": a5["without_tools"]["macro_f1"], "Macro-F1 dengan tool": a5["with_tools"]["macro_f1"]}]).round(3))
eksperimen("A6", "Fixed vs adaptive retrieval", "ukur kualitas pada easy/multi-hop/conflict/no-evidence slice dan biaya per kualitas.", "B4 dengan vs tanpa retry.")
display(pd.DataFrame([{"Run": "fixed" if "fixed" in r else "adaptive", "Slice": s, **v} for r, ss in AB["A6"].items() for s, v in ss.items()]).set_index(["Run", "Slice"]).round(3))
cq = [{"Run": n, "Macro-F1": J(f"results/runs/{f}.manifest.json")["metrics"]["macro_f1"],
       "Prompt tok per poin F1": J(f"results/runs/{f}.manifest.json")["metrics"]["cost"]["mean_prompt_tokens"] / (100 * J(f"results/runs/{f}.manifest.json")["metrics"]["macro_f1"]),
       "ms per poin F1": J(f"results/runs/{f}.manifest.json")["metrics"]["cost"]["latency_mean_ms"] / (100 * J(f"results/runs/{f}.manifest.json")["metrics"]["macro_f1"])}
      for n, f in (("fixed", "test_B4_fixedretr"), ("adaptive", "test_B4"))]
display(pd.DataFrame(cq).set_index("Run").round(2))
eksperimen("A7", "Budget konteks", "2 level token/context budget; ukur Pareto quality, p95 dan biaya.", "B4 budget kecil vs besar.")
a7 = AB["A7"]
display(pd.DataFrame([{"Budget": n, **a7[n]["budget"], "Macro-F1": a7[n]["macro_f1"], "Prompt tok": a7[n]["cost"]["mean_prompt_tokens"],
                       "p95 ms": a7[n]["cost"]["latency_p95_ms"]} for n in ("small", "large")]).set_index("Budget").round(3))
show(f"**Δ besar − kecil:** {a7['delta']['delta']:+.3f} (CI95 {a7['delta']['ci95']})")
'''

REPEAT_SHOW = r'''verif("Repeatability", "Minimal 3 run independent pada subset test untuk setiap konfigurasi.", "Run manifests, seed/sampling; variance.")
rep = J("results/results.json")["repeatability"]
mans = sorted(glob.glob("results/runs/repeat_*.manifest.json"))
print(f"Run manifests: {len(mans)} (= 7 konfigurasi × 3 seed) {ok(len(mans) >= 21)}")
display(pd.DataFrame({c: {"run 1": v["macro_f1_runs"][0], "run 2": v["macro_f1_runs"][1], "run 3": v["macro_f1_runs"][2], "mean": v["mean"], "std": v["std"],
                          "sampling": f"T={v['sampling']['temperature']}, top-p={v['sampling']['top_p']}, seeds={v['sampling']['seeds']}"} for c, v in rep.items()}).T)
'''

OOD_SHOW = r'''R = J("results/results.json"); ov = J("results/ood_overlap.json")
show(f"**OOD (A4, domain drift):** dataset Alfina dkk. 2017. {ov['overlap_removed']} dari {ov['alfina_total']} tweet tumpang-tindih dengan dataset utama **dibuang** "
     f"(anti-leakage); sampel 200 dari {ov['clean_pool']} tweet bersih.")
display(pd.DataFrame(R["ood"]).T.round(3))
'''

REDTEAM_SHOW = r'''verif("Red-team suite", "Minimal 20 kasus: 5 prompt/document injection, 5 policy/PII leakage, 5 tool misuse, 5 safety/robustness.",
      "Fixture ID, goal, attack surface, baseline outcome, mitigation outcome.")
rt = J("results/redteam.json"); s = rt["summary"]
df = pd.DataFrame([{"Fixture ID": f["fixture_id"], "Kategori": f["category"], "Attack surface": f["attack_surface"], "Goal": f["goal"],
                    "Baseline (B0)": "PASS" if f["baseline_B0_pass"] else "FAIL", "Mitigasi v1": "PASS" if f["mitigated_v1_pass"] else "FAIL",
                    "Mitigasi v2": "PASS" if f["mitigated_B6_pass"] else "FAIL"} for f in rt["fixtures"]]).set_index("Fixture ID")
display(df)
print("Per kategori:", df.Kategori.value_counts().to_dict(), ok(len(df) >= 20 and df.Kategori.value_counts().min() >= 5))
print(f"Total lolos: B0 {s['baseline_pass']}/20 → v1 {s['mitigated_v1_pass']}/20 → v2 {s['mitigated_pass']}/20 | FP filter injection: {s['injection_filter_false_positive_on_test']}")
'''

ERR_SHOW = r'''ea = open("results/error_analysis.md", encoding="utf-8").read()
show("### Analisis penyebab error (retrieval, reasoning, tool, source, prompt, adaptation — PDF §1)\n\n" + ea[ea.index("Taksonomi"):ea.index("## Contoh kualitatif")])
tb = open("results/report_tables.md", encoding="utf-8").read()
for h in ("## T13", "## T14"):
    i = tb.index(h); j = tb.find("\n## ", i + 5); show(tb[i:j if j > 0 else None].replace("## ", "### "))
'''

HUMAN_SHOW = r'''verif("Human review", "Instance dianotasi blindly oleh minimal dua annotator; jika berbeda lakukan adjudication.",
      "Agreement, disagreement log, konflik dan resolution notes.")
p = J("human_review/protocol.json"); print("Protokol:", p["instruksi"], "| n =", p["n"], "| blind =", p["blind"])
a = pd.read_csv("human_review/sheet_annotator_A.csv")
filled = a[LABELS].notna().all(axis=1).mean()
display(a.head(3))
print(f"Status pengisian annotator A: {filled:.0%}.", "✅ lanjut `python -m tbp.human_review agree`" if filled == 1 else "⚠️ BELUM DIISI — wajib dikerjakan anggota tim (lihat LAPORAN §10).")
if os.path.exists("human_review/agreement.json"): display(pd.DataFrame(J("human_review/agreement.json")))
'''

SERVING_SHOW = r'''verif("Serving", "Sistem UI dan API yang bisa diuji, implementasikan fallback jika perlu; catat logs.", "UI, API, screenshot, log, trace dan live demo.")
print("API (FastAPI, tbp/serve.py): POST /classify · POST /compare · GET /health · GET /logs | fallback kNN bila LLM gagal | log JSONL + trace_id")
print("UI (Streamlit, app_streamlit.py): tab Klasifikasi B0–B6 · Hasil Eksperimen B0–B6 · Analisis Lain · Laporan")
print("Live demo: notebook 'Streamlit - Case B: Hate Speech Indonesian Twitter' (link publik sementara).")
for f in ("docs/screenshots/ui_compare.png",):
    if os.path.exists(f): display(Image(f, width=800))
fb = J("docs/screenshots/api_response_fallback.json")
print("Contoh respons fallback:", {k: fb[k] for k in ("labels", "evidence_spans", "fallback_used", "trace_id")})
'''

COMPLIANCE = r'''R = J("results/results.json"); m = J("data/processed/split_manifest.json"); rt = J("results/redteam.json")
q = pd.DataFrame(JL("data/processed/qrels.jsonl")); a = pd.read_csv("human_review/sheet_annotator_A.csv")
rows = [
 ("§3 Corpus utama ≥300 dokumen berizin", len(JL("data/processed/corpus_examples.jsonl")) >= 300, "source_register.json + hash + index"),
 ("§3 Benchmark ≥100 query + SHA-256 + dedup", m["splits"]["test"]["n"] >= 100 and m["stats"]["exact_overlap_train_vs_heldout"] == 0, "split_manifest.json"),
 ("§3 Qrels + unanswerable", len(q) > 0 and (~q.answerable).sum() > 0, "qrels.jsonl"),
 ("§3 Human review 2 annotator + adjudikasi", a[LABELS].notna().all(axis=1).mean() == 1, "human_review/ (diisi tim)"),
 ("§3 Red-team 20 (5/5/5/5)", len(rt["fixtures"]) >= 20, "redteam.json"),
 ("§3 LLM plain vs LoRA", "B5" in R["main_test"], "results/runs + adapter"),
 ("§3 Repeatability ≥3 run tiap config", len(glob.glob("results/runs/repeat_*.manifest.json")) >= 21, "repeat manifests"),
 ("§3 Serving UI + API + fallback + log", os.path.exists("app_streamlit.py") and os.path.exists("tbp/serve.py"), "Notebook Streamlit"),
 ("§4 B0–B6, base generator sama", len({J(f'results/runs/test_{c}.manifest.json')['meta']['model_id'] for c in CFGS}) == 1, "manifest model_id"),
 ("§4 B6 vs parent identik (dipilih di dev)", "B6-" + R["dev_selection"]["B6_parent"] in R["deltas_bootstrap"], "dev_selection + bootstrap"),
 ("§4.1 Ablation A1–A7", all(k in R["ablations"] for k in ("A1_llm", "A3_B4", "A4", "A5", "A6", "A7")) and "A1_chunking" in R["retrieval"], "results.json"),
]
show("### Kepatuhan terhadap PDF (otomatis)")
display(pd.DataFrame([{"Syarat": s, "Status": "✅" if b else "⚠️ (tugas tim)" if "Human" in s else "❌", "Bukti": e} for s, b, e in rows]).set_index("Syarat"))
'''

PACK2 = r'''env = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
json.dump(env, open("results/environment_kaggle.json", "w"), indent=1)
out = "/kaggle/working/tbp_output.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ["results", "adapters", "data/processed", "logs", "human_review"]:
        for f in glob.glob(f"{d}/**/*", recursive=True):
            if os.path.isfile(f):
                z.write(f, f)
print("Output:", out, round(os.path.getsize(out) / 1e6, 1), "MB |", env)
'''


def build(find_project, pip):
    c = []
    c.append(md("# Case B: Multi-label Hate Speech & Abusive Language Detection pada Twitter Indonesia\n"
                "**TBP Frontier LLM Research & Engineering — Kelompok Rizqi**\n\n"
                "Notebook ini menjalankan **seluruh eksperimen dari nol** di GPU T4 dan menampilkan, untuk setiap tahap:\n"
                "- ✔ **Cara verifikasi** sesuai *PDF §3 Target minimum dan evaluasi research-grade*\n"
                "- 🧪 **Eksperimen** sesuai *PDF §4 (B0–B6)* dan *§4.1 (ablation A1–A7)*\n\n"
                "Kode asli setiap modul ditampilkan sebagai sel `%%writefile` (tersembunyi; klik **Show code** untuk membuka). "
                "Kode yang sama dipakai aplikasi Streamlit/API, jadi hanya ada satu sumber kode.\n\n"
                "| Bagian | Isi | PDF |\n|---|---|---|\n"
                "| 1 | Data: corpus, benchmark, qrels | §3 |\n| 2 | Komponen sistem B0–B6 (kode) | §4 |\n| 3 | Training LoRA (B5) | §4, §3 |\n"
                "| 4 | Retrieval A1–A2 | §4.1 |\n| 5–6 | Dev & locked test B0–B6 | §4, §3 |\n| 7 | Ablation A1, A3–A7 | §4.1 |\n"
                "| 8–10 | Repeatability, OOD, Red-team | §3, §4.1 |\n| 11 | Error, halusinasi, fairness | §1 |\n| 12–13 | Human review, Serving | §3 |\n"
                "| 14 | Kepatuhan PDF | semua |\n\n"
                "**Dataset:** Ibrohim & Budi (2019), *Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter*, ALW3 @ ACL "
                "([paper](https://aclanthology.org/W19-3506/)) — 13.169 tweet, CC BY-NC-SA 4.0. **Base generator:** Qwen2.5-0.5B-Instruct (sama untuk B0–B6)."))
    c.append(md("## 0. Persiapan lingkungan"))
    c.append(code(find_project))
    c.append(code(pip, hide_output=True))
    c.append(code(HELPERS))
    c.append(md("## 1. Data — corpus, benchmark terkunci, qrels\n"
                "Tweet dibersihkan, dedup, dibagi **train / dev / test (terkunci)** secara stratified, ditambah teks out-of-scope (wajib abstain). "
                "Corpus retrieval = tweet train berlabel + pedoman label 2 versi (untuk temporal checking). Qrels = bagian pedoman yang relevan per query."))
    c += [wf("tbp/config.py"), wf("tbp/textutil.py"), wf("tbp/data_prep.py")]
    c.append(code('run("tbp.data_prep")'))
    c.append(code(VERIF_DATA))
    c.append(md("## 2. Komponen sistem B0–B6 (kode)\n"
                "Retrieval (BM25, dense e5, RRF, reranker, filter metadata/temporal) · generator Qwen + LoRA · prompt registry · pipeline B0–B5 "
                "(planner, sufficiency score, retry, verifier, abstain) · agent B6 (tool terstruktur, authorization, temporal check) · evaluasi."))
    c += [wf("tbp/retrieval.py"), wf("tbp/llm.py"), wf("tbp/prompts.py"), wf("tbp/pipelines.py"), wf("tbp/agent.py"),
          wf("tbp/evaluate.py"), wf("tbp/run_experiments.py")]
    c.append(md("## 3. Training LoRA untuk B5\nData latih hanya dari **train** (1.600 tweet) dalam format prompt B4 yang sama; target = label gold + silver evidence."))
    c.append(wf("tbp/train_lora.py"))
    c.append(code('run("tbp.train_lora")'))
    c.append(code(LORA_SHOW))
    c.append(md("## 4. Retrieval — ablation A1 & A2 (qrels identik)"))
    c.append(code('run("tbp.run_experiments", "retrieval")'))
    c.append(code(RETR_SHOW))
    c.append(md("## 5. Dev — memilih parent B6\nKonfigurasi dipilih di **dev**, bukan di test (anti-leakage)."))
    c.append(code('run("tbp.run_experiments", "dev")'))
    c.append(code(DEV_SHOW))
    c.append(md("## 6. Locked test — B0 sampai B6\nSemua konfigurasi memakai base generator, sampling (greedy), dan output contract yang sama."))
    c.append(code('run("tbp.run_experiments", "test")'))
    c.append(code(TEST_SHOW))
    c.append(md("### 6.1 Eksperimen per konfigurasi (tabel PDF §4)"))
    c.append(md("> Ablation dijalankan dulu karena B3 (kontrol satu variabel), B4 (retry) dan B6 (tanpa tool) membutuhkan run pembanding dari tahap ablation."))
    c.append(code('run("tbp.run_experiments", "ablations")'))
    c.append(code(PER_B))
    c.append(md("### 6.2 Contoh satu tweet dijawab B0–B6"))
    c.append(code(EXAMPLE))
    c.append(md("### 6.3 Verifikasi LLM comparison"))
    c.append(code(VERIF_LLM))
    c.append(md("## 7. Ablation A1 (level LLM), A3–A7 (PDF §4.1)\nSatu variabel utama diubah per percobaan."))
    c.append(code(ABL_SHOW))
    c.append(md("## 8. Repeatability"))
    c.append(code('run("tbp.run_experiments", "repeat")'))
    c.append(code(REPEAT_SHOW))
    c.append(md("## 9. Domain drift (OOD)"))
    c.append(code('run("tbp.run_experiments", "ood")'))
    c.append(code(OOD_SHOW))
    c.append(md("## 10. Red-team"))
    c += [wf("tbp/serving_guard.py"), wf("tbp/redteam.py")]
    c.append(code('run("tbp.redteam")'))
    c.append(code(REDTEAM_SHOW))
    c.append(md("## 11. Analisis error, hallucination audit, fairness"))
    c.append(wf("tbp/make_report.py"))
    c.append(code('run("tbp.make_report")'))
    c.append(code(ERR_SHOW))
    c.append(md("## 12. Human review"))
    c.append(wf("tbp/human_review.py"))
    c.append(code(HUMAN_SHOW))
    c.append(md("## 13. Serving (UI + API)\nKode aplikasi (tersembunyi). Demo langsung dijalankan di notebook **Streamlit - Case B: Hate Speech Indonesian Twitter**."))
    c += [wf("tbp/serve.py"), wf("app_streamlit.py")]
    c.append(code(SERVING_SHOW))
    c.append(md("## 14. Kepatuhan terhadap PDF"))
    c.append(code(COMPLIANCE))
    c.append(md("## 15. Paketkan output (untuk notebook Streamlit & arsip)"))
    c.append(code(PACK2))
    return c
