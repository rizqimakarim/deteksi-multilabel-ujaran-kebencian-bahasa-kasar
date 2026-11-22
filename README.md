# TBP Kelompok Rizqi — Evidence-Aware Multi-label Hate Speech & Abusive Language Detection (Case B)

Sistem klasifikasi **multi-label** tweet Indonesia informal (4 label biner: `HS`, `Abusive`, `HS_Individual`, `HS_Group`) dengan rationale span teks, abstain, retrieval pedoman label + contoh train, LoRA, agentic tool use, dan evaluation harness lengkap (B0–B6, ablation A1–A7, red-team, repeatability, OOD).

🧭 **Baru ikut proyek? Mulai dari [ALUR_END_TO_END.md](ALUR_END_TO_END.md)** (urutan kerja dari unduh dataset sampai dashboard), lalu [RINGKASAN_PROYEK.md](RINGKASAN_PROYEK.md).

📄 **Laporan lengkap:** [LAPORAN.md](LAPORAN.md) · **ADR:** [docs/ADR.md](docs/ADR.md) · **Tabel hasil:** [results/report_tables.md](results/report_tables.md)

## Struktur
```
data/raw/                 dataset Ibrohim & Budi 2019 + kamus alay + lexicon (CC BY-NC-SA 4.0)
data/ood/                 Alfina et al. 2017 (OOD; JANGAN dipublikasikan ulang)
data/guidelines/          pedoman label v1.0 (berlaku) & v0.9 (usang) -> corpus retrieval + temporal check
data/processed/           split train/dev/test (+ split_manifest.json SHA-256), corpus, chunk, qrels, lexicon
data/source_register.json sumber, lisensi, hash, penggunaan
tbp/                      kode (config, data_prep, retrieval, llm, prompts, pipelines, agent, train_lora,
                          evaluate, run_experiments, redteam, serve, serving_guard, human_review, make_report)
adapters/lora_b5/         adapter LoRA + training log
results/runs/             raw output per item + manifest per run (prompt/model/index version, token, latency)
results/results.json      semua metrik; results/report_tables.md tabel laporan; results/redteam.json
human_review/             sheet anotasi blind A/B + protokol (diisi anggota tim)
logs/                     log tiap tahap + serving.jsonl (observability)
```

## Cara menjalankan
```bash
# 1. environment (Windows, GPU NVIDIA; di Linux/Kaggle cukup `pip install`)
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install transformers peft accelerate sentence-transformers rank_bm25 fastapi uvicorn scikit-learn pandas

# 2. seluruh eksperimen (data -> LoRA -> B0..B6 -> ablation -> repeat -> OOD -> red-team -> tabel)
bash run_all.sh

# 3. demo: dobel-klik start_demo.bat  (API FastAPI :8000 + UI Streamlit :8501)
#    atau manual:
.venv/Scripts/python -m uvicorn tbp.serve:app --port 8000
.venv/Scripts/python -m streamlit run app_streamlit.py --server.port 8501
#   UI Streamlit http://localhost:8501 -> tab Klasifikasi (B0-B6 berdampingan) | Hasil Eksperimen | Laporan
#   buka http://127.0.0.1:8000  (UI: satu tombol, hasil B0-B6 berdampingan)
#   API: POST /classify {"text":"...","config":"B6"} | POST /compare {"text":"..."} | GET /logs | GET /health
```
Contoh API:
```bash
curl -X POST http://127.0.0.1:8000/classify -H "Content-Type: application/json" \
     -d '{"text":"USER dasar goblok lu, cebong semua sama aja","config":"B6"}'
```

## Menjalankan di Kaggle (GPU T4) — cara utama untuk tim
Semua eksperimen resmi dijalankan di Kaggle agar anggota tim tanpa GPU bisa memakai proyek ini. Isi di akun `rizqimakarim` (private):
- Dataset **Case B: Multi-label Hate Speech Indonesian Twitter** — kode + data.
- Notebook **Case B: Multilabel Hate Speech Indonesian Twitter** — menjalankan seluruh eksperimen (±2–3 jam).
- Notebook **Streamlit - Case B: Hate Speech Indonesian Twitter** — demo aplikasi dengan link publik sementara.

Langkah lengkap: **[KAGGLE_PANDUAN.md](KAGGLE_PANDUAN.md)**. Paket diunggah ulang dengan `python tools/make_kaggle_package.py <username>`.

## Human review (WAJIB dikerjakan anggota tim)
```bash
python -m tbp.human_review make       # sudah dibuat: human_review/sheet_annotator_A.csv & _B.csv
# annotator A & B mengisi sheet masing-masing secara terpisah (blind)
python -m tbp.human_review agree      # kappa + disagreement_log + adjudication_sheet
# adjudikator mengisi adjudication_sheet.csv
python -m tbp.human_review finalize   # human_gold + evaluasi model terhadap human gold
```

## Sumber & lisensi
- Ibrohim, M.O. & Budi, I. (2019). *Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter*. ALW3. CC BY-NC-SA 4.0.
- Alfina, I. et al. (2017). *Hate Speech Detection in the Indonesian Language: A Dataset and Preliminary Study*. ICACSIS (dipakai lokal untuk OOD, tidak diredistribusi).
- Model: Qwen2.5-0.5B-Instruct (Apache-2.0), multilingual-e5-small (MIT), mmarco-mMiniLMv2 cross-encoder (Apache-2.0).
