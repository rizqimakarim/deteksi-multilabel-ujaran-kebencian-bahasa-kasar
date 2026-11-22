# Architecture Decision Records (ADR)

Format: Konteks → Keputusan → Alasan → Konsekuensi. Status semua ADR: **Diterima**.

## ADR-001 Use case B varian multi-label ujaran kebencian/bahasa kasar
- **Konteks:** Use case B (sentimen & emosi) membingungkan karena satu kalimat bisa positif dan negatif sekaligus. Dosen (chat 18/11/2026) mengizinkan multilabel lain dengan contoh *toxic/insult/threat/obscene*, yaitu label biner independen per teks.
- **Keputusan:** Task = klasifikasi multi-label 4 label biner pada tweet Indonesia informal/campur bahasa: `HS`, `Abusive`, `HS_Individual`, `HS_Group` (skema Ibrohim & Budi, 2019).
- **Alasan:** Setiap label ya/tidak secara independen, jadi tidak ambigu di level teks (mengikuti arahan dosen). Dataset publik berlisensi CC BY-NC-SA 4.0, berbahasa gaul, dan punya ambiguitas multi-label nyata.
- **Konsekuensi:** Metrik = macro-F1, per-label F1, micro-F1, exact match, dan hamming loss. Label `HS_Weak/Moderate/Strong` dan sub-target (agama/ras/…) tidak dipakai agar kelas tidak terlalu jarang (<7%).

## ADR-002 Base generator Qwen2.5-0.5B-Instruct (sama untuk B0–B6)
- **Konteks:** GPU lokal RTX 2050 4 GB. Spek: compact open-weight 0.5B–3B, dengan base generator sama di B0–B5.
- **Keputusan:** `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0), bf16.
- **Alasan:** Muat untuk inference sekaligus LoRA training di 4 GB. Model ini multibahasa (termasuk Indonesia) dan mendukung chat template.
- **Konsekuensi:** Kemampuan zero-shot rendah (B0 lemah). Peningkatan diukur relatif terhadap model yang sama. Untuk eksperimen lanjutan, model dapat diganti 1.5B/3B di `tbp/config.py` (Kaggle/Colab T4).

## ADR-003 LoRA bf16 (bukan QLoRA) — SETARA
- **Konteks:** Spek menyebut LoRA/QLoRA. QLoRA butuh `bitsandbytes`, yang dukungannya di Windows terbatas.
- **Keputusan:** LoRA r=16, α=32, dropout 0.05, pada semua proyeksi attention + MLP, dengan base bf16.
- **Alasan:** Metodenya setara (adapter low-rank, base dibekukan). Untuk model 0.5B, kuantisasi 4-bit tidak diperlukan karena memori cukup. Hasilnya lebih stabil dan reproducible.
- **Konsekuensi:** Bukti konfigurasi & resource ada di `adapters/lora_b5_training_log.json`.

## ADR-004 Retriever: rank_bm25 + multilingual-e5-small + RRF + mMiniLM cross-encoder
- **Keputusan:** BM25 (`rank_bm25`, token dinormalisasi kamus alay), dense `intfloat/multilingual-e5-small`, fusion Reciprocal Rank Fusion (k=60), dan reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Index berupa numpy in-memory (bukan FAISS/vector DB).
- **Alasan:** Corpus ±12,6 ribu dokumen pendek muat di memori. Brute-force cosine itu exact, dan FAISS tidak menambah kemampuan pada ukuran ini (SETARA). Semua model multibahasa dan kecil.
- **Konsekuensi:** Versi index = hash isi dokumen (dicatat di setiap run manifest).

## ADR-005 Corpus = contoh berlabel TRAIN + pedoman label berversi
- **Konteks:** Spek use case B: "gunakan retrieval label policy dan contoh berizin dari train split".
- **Keputusan:** Corpus terdiri dari 12.594 tweet train (doc_type=example) ditambah dokumen pedoman label v1.0 (berlaku mulai 2026-07-01) dan v0.9 (usang, berlaku 2025-01-01 s/d 2026-06-30). Pedoman dipecah dengan 3 strategi chunking (A1).
- **Alasan:** Retrieval relevan terhadap jawaban (contoh mirip + definisi label). Dua versi pedoman memungkinkan temporal checking dan memberi uji injeksi dokumen usang.
- **Konsekuensi:** Filter metadata wajib: `split=train` (anti-leakage) dan `valid_from ≤ as_of ≤ valid_to`. B2 (naive) sengaja tanpa filter.

## ADR-006 Qrels level section pedoman
- **Konteks:** Task ini klasifikasi, bukan QA, sehingga tidak ada "dokumen jawaban" tunggal.
- **Keputusan (aturan anotasi R1):** Untuk setiap query test/dev, section pedoman yang relevan adalah section milik label positif gold (rel=2). Teks netral → section NONE (rel=1). Teks out-of-scope → section SCOPE/abstain (rel=2, `answerable=false`). Chunk dianggap relevan bila mencakup section tersebut.
- **Alasan:** Qrels turun dari label gold secara deterministik dan dapat diaudit. Recall/nDCG/MRR dapat dihitung identik untuk semua retriever (A2).
- **Tambahan:** Kualitas index contoh diukur dengan *label precision@5* (tetangga dengan vektor label identik).

## ADR-007 Pemetaan slice A6 untuk task klasifikasi
- easy = 0–1 label positif; *multi-hop* → **multi** = ≥2 label positif (keputusan berantai: HS → target); **conflict** = mayoritas label 5 tetangga BM25 ≠ gold (bukti corpus bertentangan); **no-evidence** = out-of-scope. Definisi dihitung dari gold + corpus, bukan dari prediksi.

## ADR-008 Abstain dihitung sebagai prediksi negatif
- Untuk macro-F1 utama, item in-scope yang di-abstain dihitung "semua 0" (konservatif, tidak menguntungkan abstain). Selain itu dilaporkan *selective macro-F1* (hanya item terjawab), coverage/false-refusal, dan OOS-abstain recall.

## ADR-009 Silver rationale untuk target LoRA
- Dataset tidak punya span rationale. Target `evidence` saat training = kata pada teks yang termasuk lexicon chi² per label (dihitung **hanya dari train**) ∪ `abusive.csv`. Ini silver label dan dicatat sebagai keterbatasan. Evaluasi evidence = span harus verbatim di teks (grounded), bukan dicocokkan dengan silver.

## ADR-010 Serving: FastAPI (API) + Streamlit (UI) — SETARA
- **Pembaruan:** UI utama dipindah ke Streamlit (3 tab: klasifikasi B0–B6 berdampingan, hasil eksperimen, laporan) agar demo dan hasil eksperimen ada dalam satu aplikasi. Streamlit hanya lapisan tampilan; logika tetap di API FastAPI (`/classify`, `/compare`), sehingga UI dan API tetap terpisah dan bisa diuji sendiri-sendiri. Halaman HTML di `GET /` dipertahankan sebagai UI cadangan.
- Satu proses melayani API (`/classify`, `/health`, `/logs`) dan UI. Observability: log JSONL per request (trace_id, config, label, skor, span, versi pedoman, token, latensi, fallback). Fallback = voting kNN contoh train bila LLM gagal atau output tidak valid.

## ADR-011 Data OOD tidak didistribusikan ulang
- Dataset Alfina et al. (2017) boleh dipakai dengan sitasi tetapi **dilarang redistribusi publik**. Karena dataset utama (Ibrohim & Budi 2019) memuat sebagian tweet Alfina, tweet OOD yang sama/near-duplicate (cosine ≥ 0,8) dengan dataset utama dibuang (193 dari 713) sebelum evaluasi. File hanya dipakai lokal untuk evaluasi OOD (A4) dan tidak boleh di-commit ke repo publik (lihat `.gitignore`).

## ADR-012 Human review oleh anggota tim
- Anotasi blind 2 annotator wajib dilakukan manusia, sehingga **tidak disimulasikan dengan model**. Sistem menyediakan sheet blind (urutan acak berbeda, tanpa gold/prediksi), script kappa, disagreement log, dan adjudikasi (`tbp/human_review.py`).
