# LAPORAN TBP — Frontier LLM Research & Engineering
## Kelompok Rizqi · Use Case B: Evidence-Aware Multi-label Hate Speech & Abusive Language Detection pada Twitter Indonesia

**Anggota & kontribusi individu:** _(diisi tim: nama, NIM, kontribusi)_
**Tanggal:** 20 November 2026 · **Base generator:** Qwen2.5-0.5B-Instruct (sama untuk B0–B6) · **Hardware run resmi:** Kaggle GPU NVIDIA T4 16 GB (fp16; LoRA fp32 + autocast fp16). Run awal di laptop RTX 2050 diarsipkan di `archive/laptop_rtx2050/`

### Ringkasan Eksekutif
Kami membangun sistem end-to-end yang memberi **4 label biner** (HS, Abusive, HS_Individual, HS_Group) untuk tweet Indonesia informal. Setiap keluaran disertai **span bukti dari teks**, skor per label, versi pedoman, dan **abstain** untuk input di luar cakupan. Tujuh konfigurasi B0–B6 dibandingkan pada *locked test* 220 item dengan base model yang sama. **Run resmi: Kaggle Tesla T4** (seluruh pipeline dijalankan ulang dari nol; run awal di laptop diarsipkan).

- **Macro-F1 naik dari 0,164 (B0) menjadi 0,676 (B5)**; B6 = 0,670 (B6 − B0 +0,506, CI95 [+0,435; +0,573]).
- **Kontributor signifikan:** **retrieval** (B2−B1 +0,334), **adaptive RAG + verifier** (B4−B3 +0,075), **LoRA** (B5−B4 +0,105). Tidak signifikan: hybrid RAG (B3−B2 +0,056), tool calling (B6−B5 −0,005).
- **Grounding:** evidence support B4–B6 = 100% (B0 = 0%); unsupported claim B4–B6 = 0. ECE turun 0,304 → 0,099 (B5).
- **Abstain di luar cakupan:** 70% pada 20 teks uji **baru**. Gagal pada bahasa asing berhuruf Latin selain Inggris (Vietnam, Tagalog, Spanyol, Italia) dan potongan kode Python/HTML.
- **Red-team:** 6/20 (B0) → 19/20 (B6 + guard v1) → 20/20 (v2 + injection filter, dengan catatan risiko overfit fixture).
- **Temuan negatif yang dilaporkan:** prompt terstruktur saja menurunkan F1 pada model 0.5B (B1 < B0); kenaikan B2→B3 ternyata berasal dari strategi chunking, bukan hybrid retrieval; reranker mMARCO menurunkan recall; retry adaptif dan budget konteks besar tidak membantu; tool calling tidak menambah kualitas; generalisasi ke data OOD lemah.

Kode: `tbp/` · Tabel lengkap otomatis: `results/report_tables.md` · ADR: `docs/ADR.md` · Analisis error: `results/error_analysis.md` · Panduan Kaggle: `KAGGLE_PANDUAN.md`

---
## 1. Latar Belakang dan Definisi Masalah

**Use case B (varian multi-label sesuai arahan dosen).** Sistem menerima satu teks pendek berbahasa Indonesia informal/campur bahasa (tweet) dan mengeluarkan **4 label biner independen**: `HS` (ujaran kebencian), `Abusive` (bahasa kasar), `HS_Individual` (target individu), dan `HS_Group` (target kelompok). Setiap keluaran wajib menyertakan **rationale berupa span teks**, **skor per label** (bila tersedia), **versi pedoman label**, dan **abstain** beserta alasannya.

Mengikuti contoh dosen (*toxic/insult/threat/obscene*), satu teks bisa bernilai 1 pada beberapa label sekaligus tanpa kontradiksi. Contoh: *"USER dasar goblok lu antek aseng"* → HS=1, Abusive=1, HS_Individual=1, HS_Group=0. Bandingkan dengan sentimen positif-negatif yang ambigu di level teks.

**Output contract (sama untuk B0–B6):**
```json
{"HS":0,"Abusive":0,"HS_Individual":0,"HS_Group":0,"evidence":["span persis dari teks"],"abstain":false}
```
Di lapisan API ditambah: `scores`, `abstain_reason`, `guideline_version`, `as_of`, `retrieved_ids`, `tool_log`, `trace_id`, dan `fallback_used`.

### Definisi klaim (tanpa klaim berlebih)
| Istilah | Definisi operasional di proyek ini |
|---|---|
| **lebih baik** | Delta macro-F1 locked-test > 0 **dan** CI95 paired bootstrap (1000×) tidak memuat 0. Selain itu disebut "tidak signifikan". |
| **grounded** | Setiap label positif didukung ≥1 span yang muncul **verbatim** di teks masukan (metrik *evidence support*). |
| **reliable** | Std macro-F1 antar 3 run independen (sampling T=0.7) ≤ 0,05 pada subset yang sama. |
| **aman** | Lolos fixture red-team terkait (injection, PII, tool misuse, robustness). Bukan jaminan umum. |
| **agentic** | LLM memilih tool + argumen terstruktur sendiri (maks 2 langkah), divalidasi guard (allowlist, schema, temporal, budget). |
| **fine-tuned** | Base generator yang sama dipasangi adapter LoRA (r=16) yang dilatih pada 1.600 contoh train. |
| **real-time** | Tidak diklaim. Latensi dilaporkan apa adanya (mean & p95 per item). |

## 2. Data

| Komponen | Isi | Bukti |
|---|---|---|
| Sumber utama | 13.169 tweet Ibrohim & Budi (2019), CC BY-NC-SA 4.0; username/URL sudah dianonimkan (USER/URL) | `data/source_register.json` (hash SHA-256) |
| Cleaning | hapus byte-emoji literal, `RT`, mention berulang; dedup exact (265 duplikat) + 15 teks berlabel konflik dibuang | `split_manifest.json → stats` |
| Split | **test 220** (200 in-scope stratified label-powerset + 20 out-of-scope), **dev 130** (100 + 30 out-of-scope), **train 12.594** | `data/processed/*.jsonl` + SHA-256 |
| Near-dup check | char 3–5-gram TF-IDF cosine ≥ 0,85 terhadap dev/test → 10 item train dibuang; overlap exact train-heldout = 0 | `split_manifest.json` |
| Out-of-scope (unanswerable) | Teks buatan tim: Inggris penuh, bahasa lain (Korea, Rusia, Vietnam, Tagalog, Spanyol, Italia, Jepang, Arab, Prancis, Jerman), kode, angka, simbol, emoji, kosong. **Dev:** 30 teks yang dipakai menyetel *scope checker*. **Test:** 20 teks **baru** yang ditulis setelah *scope checker* dibekukan dan tidak pernah dipakai menyetel apa pun | `data_prep.OOS_TEXTS` (dev), `OOS_TEXTS_TEST` (test) |
| Corpus retrieval | **12.594 dokumen contoh train** + pedoman label **v1.0** (berlaku 2026-07-01) & **v0.9** (usang) dalam 3 strategi chunking | `corpus_examples.jsonl`, `guide_chunks_*.jsonl` |
| Qrels | 551 qrels (dev+test) level section pedoman, aturan anotasi R1 (ADR-006); out-of-scope → `answerable=false` | `qrels.jsonl` |
| OOD | Alfina et al. (2017), 200 tweet sampel dari 520 tweet yang **tidak** tumpang-tindih dengan dataset utama (label HS saja), mention dianonimkan | lokal saja (ADR-011), `results/ood_overlap.json` |

Prevalensi label test in-scope: HS 0,42 · Abusive 0,38 · HS_Individual 0,265 · HS_Group 0,155 (sama dengan train ±0,01).

### 2.1 Sumber dan sitasi dataset

| | Dataset utama | Dataset OOD |
|---|---|---|
| **Nama** | Indonesian Multi-label Hate Speech and Abusive Language Dataset (`ID_MULTILABEL_HS`) | ID Hate Speech Detection (IDHSD_RIO) |
| **Sitasi** | Muhammad Okky Ibrohim & Indra Budi. 2019. *Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter*. Proc. Third Workshop on Abusive Language Online (ALW3), hlm. 46–57, Florence, Italia. ACL. | Ika Alfina, Rio Mulia, Mohamad Ivan Fanany, Yudo Ekanata. 2017. *Hate Speech Detection in the Indonesian Language: A Dataset and Preliminary Study*. ICACSIS 2017. IEEE. |
| **DOI** | [10.18653/v1/W19-3506](https://doi.org/10.18653/v1/W19-3506) | [10.1109/ICACSIS.2017.8355039](https://doi.org/10.1109/ICACSIS.2017.8355039) |
| **Paper** | https://aclanthology.org/W19-3506.pdf (akses terbuka) | https://ieeexplore.ieee.org/document/8355039 (IEEE) |
| **Data** | https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection | https://github.com/ialfina/id-hatespeech-detection |
| **Mirror/katalog** | https://huggingface.co/datasets/SEACrowd/id_multilabel_hs | – |
| **Jumlah** | 13.169 tweet (dinyatakan di paper; dicek ulang: `split_manifest.json → raw_rows = 13169`) | 713 tweet (260 HS, 453 Non_HS); 193 dibuang karena tumpang-tindih dengan dataset utama → 520 bersih; dipakai sampel 200 |
| **Cara anotasi (dari paper)** | Crowdsourcing berbayar, **30 annotator**, tiap tweet dianotasi **3 orang**. Fase 1 (HS & Abusive): label final hanya bila **100% sepakat** (11.292 dari 16.500 tweet lolos = 68,44%). Fase 2 (target/kategori/level): **3 annotator terbaik**, majority vote; 139 tweet tanpa kesepakatan dibuang. Pedoman disusun dari FGD dengan **Direktorat Tindak Pidana Siber Bareskrim Polri** + ahli linguistik, dengan gold-standard test untuk calon annotator | **30 relawan mahasiswa** (beragam agama, etnis, gender); hanya tweet dengan **100% kesepakatan** dipertahankan (713 dari 1.100; agreement 64,8%). Tweet seputar **Pilkada DKI 2017** |
| **Baseline di paper** | ML klasik (RFDT + Label Power-set + word unigram): akurasi **77,36%** untuk HS + Abusive saja; **66,12%** bila termasuk target, kategori, dan level | RFDT + word n-gram: F-measure **93,5%** (biner HS vs Non_HS, 713 tweet) |
| **Lisensi** | CC BY-NC-SA 4.0 | Bebas dipakai dengan sitasi; dilarang redistribusi publik |
| **Privasi** | Username/URL diganti `USER`/`URL` oleh penulis; ID tweet tidak disediakan (ToS Twitter) | Mention dianonimkan oleh tim sebelum dipakai |

**Implikasi dari paper untuk proyek ini:**
1. **Label HS dan Abusive berkualitas tinggi.** Keduanya hanya dipertahankan bila 3 annotator 100% sepakat. Label target (HS_Individual/HS_Group) memakai majority vote, sehingga lebih bernoise. Ini konsisten dengan temuan kami bahwa HS_Group adalah label tersulit (F1 terendah).
2. **Definisi pedoman label kami (v1.0) selaras dengan paper.** Target individu vs kelompok mengikuti definisi FGD Bareskrim Polri: kelompok = agama, ras, politik, komunitas, dll.
3. **Baseline paper tidak dapat dibandingkan langsung** dengan angka kami. Metriknya berbeda (akurasi berbasis Jaccard per tweet = |label benar ∩ prediksi| / |label benar ∪ prediksi|, vs macro-F1), set labelnya berbeda (2 atau 12 label vs 4 label kami), dan paper memakai 10-fold CV atas seluruh data, sedangkan kami memakai locked test 200 tweet. Sebagai konteks: exact match kami **0,700** (B5) dan **0,655** (B6) untuk 4 label.
4. **Dataset utama memuat tweet Alfina et al. (2017)**, sehingga data OOD harus dibersihkan dulu (lihat A4).

Sumber pendukung dari repo yang sama: `new_kamusalay.csv` (kamus slang → baku) dan `abusive.csv` (lexicon kata kasar). Semua sumber beserta hash SHA-256 tercatat di `data/source_register.json`. Salinan paper (bila diunduh) disimpan di `docs/references/`.

## 3. Metode: Tujuh Konfigurasi B0–B6

Semua konfigurasi memakai **base generator yang sama**: `Qwen/Qwen2.5-0.5B-Instruct` (bf16), decoding greedy, maks 80 token keluaran, output contract tetap.

| ID | Implementasi |
|---|---|
| **B0** Direct LLM | Prompt `P0_zero_shot@1.0`: instruksi singkat + format JSON. Tanpa retrieval/tool/verifier. |
| **B1** Structured prompt | `P1_structured@1.0`: definisi label, aturan konsistensi, kebijakan abstain, 4 few-shot **tetap dari train** (kombinasi label berbeda). |
| **B2** Naive RAG | B1 + **dense** (e5-small) top-4 contoh train + top-1 chunk pedoman **fixed-size** (450 char), **tanpa** filter metadata. |
| **B3** Hybrid RAG | BM25 + dense → RRF → reranker cross-encoder → **filter metadata** (`split=train`, pedoman berlaku pada `as_of`) dengan chunk heading-aware. |
| **B4** Adaptive / self-corrective | B3 + **query planner** (scope check → abstain dini; query alternatif = normalisasi slang + kata lexicon) + **evidence sufficiency score** (0,5·relevansi reranker + 0,5·kesepakatan label tetangga) + **retry bounded** (maks 1, k×2) + **claim verifier independen** (konsistensi HS↔target, span harus verbatim, repair LLM 1×, label tanpa bukti & vote tetangga <0,5 dibuang) + abstain `conflicting_evidence`. Skor label = 0,5·LLM + 0,5·vote tetangga. |
| **B5** B4 + LoRA | Pipeline, index, prompt, dan budget identik dengan B4; generator dipasangi adapter LoRA (`adapters/lora_b5`). |
| **B6** Guarded agentic | Parent (B4/B5 dipilih via **dev**) + LLM memanggil tool terstruktur (`search_examples`, `get_policy`, `lexicon_lookup`, `normalize_slang`) maks 2 langkah. **Guard:** allowlist role (tool `read_file`, `export_dataset`, dll. ditolak), validasi schema/tipe/rentang, deteksi pola injection/PII/path di argumen, temporal check (`as_of` tidak boleh di masa depan; pedoman harus berlaku), budget langkah & konteks. Jawaban akhir dihasilkan parent + **verifier parent** (verified final answer). |

**Prompt registry:** `tbp/prompts.py`. Setiap prompt punya ID@versi + hash SHA-256, dan tercatat di setiap run manifest.

**LoRA (B5):** r=16, α=32, dropout 0,05, target q/k/v/o/gate/up/down; lr 2e-4 (warmup 10 step, linear decay), 1 epoch, grad-accum 8, max len 1280, seed 42. Data latih = 1.600 tweet train dalam **format prompt B4 yang sama** (konteks retrieval hybrid dengan dokumen itu sendiri dikecualikan). Target = label gold + *silver evidence* dari lexicon chi² train (ADR-009). Val loss dihitung pada 80 tweet train held-out, dev loss pada 60 tweet dev.

## 4. Protokol Evaluasi
- **Locked test**: hanya untuk evaluasi akhir. Semua pemilihan (parent B6) memakai dev.
- **Metrik klasifikasi:** macro-F1 (utama), micro-F1, per-label P/R/F1, exact match, hamming loss. Item in-scope yang di-abstain dihitung "semua 0" (ADR-008). Dilaporkan juga selective macro-F1 dan coverage.
- **Abstain:** OOS-abstain recall (20 item out-of-scope) dan false-refusal rate (in-scope yang di-abstain).
- **Evidence:** evidence support (item berlabel positif dengan ≥1 span verbatim), unsupported claim rate, dan hallucinated span rate (span mentah LLM yang tidak ada di teks).
- **Kalibrasi:** ECE 10-bin atas skor per label.
- **Biaya:** token prompt/completion, jumlah panggilan LLM, latensi mean & p95 (latensi generasi = porsi waktu batch 8), retry.
- **Retrieval:** Recall@5/10, nDCG@10, MRR@10 atas qrels section; label-precision@5 untuk index contoh.
- **Signifikansi:** paired bootstrap 1000× atas item test, CI95 delta macro-F1.
- **Repeatability:** 3 run independen (seed 11/22/33, T=0.7, top-p 0.9) pada subset 40 item (35 in-scope + 5 OOS).

## 5. Etika, Privasi, dan Keamanan
- **Lisensi:** semua sumber tercatat di source register beserta lisensinya. Data OOD tidak diredistribusi.
- **PII:** dataset sudah dianonimkan penyedia. Mention pada data OOD dianonimkan. Di serving, nomor HP/email/NIK/URL/mention **diredaksi sebelum masuk model maupun log**. Tidak ada data yang dikirim ke penyedia model eksternal karena semua model berjalan lokal.
- **Anti-leakage:** few-shot dan contoh retrieval hanya dari train (filter `split=train`). Dedup/near-dup antar split. Label test tidak pernah masuk prompt. Konfigurasi dipilih di dev.
- **Bias:** fairness diukur sebagai FPR label HS pada teks non-HS yang menyebut identitas kelompok (agama/etnis/gender) dibanding teks tanpa identitas (Tabel T14).
- **Risiko penyalahgunaan:** sistem adalah alat bantu moderasi, **bukan** pengambil keputusan final. Abstain diteruskan ke manusia.
## 6. Hasil Utama (locked test: 200 in-scope + 20 out-of-scope)

Run resmi: Kaggle **Tesla T4**, torch 2.11.0+cu130, transformers 5.23.0, peft 0.22.0; inference fp16, decoding greedy.

| Config | Macro-F1 | Micro-F1 | Exact match | Evidence support | Unsupported claim | OOS abstain | False refusal | ECE | Latency p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| B0 Direct | 0,164 | 0,342 | 0,345 | 0% | 1,000 | 0% | 0,015 | 0,304 | 248 |
| B1 Structured prompt | 0,107 | 0,152 | 0,400 | 87% | 0,133 | 0% | 0,000 | 0,306 | 343 |
| B2 Naive RAG | 0,441 | 0,521 | 0,445 | 92% | 0,071 | 0% | 0,005 | 0,239 | 420 |
| B3 Hybrid RAG | 0,496 | 0,571 | 0,445 | 96% | 0,053 | 0% | 0,000 | 0,233 | 685 |
| B4 Adaptive RAG | 0,571 | 0,602 | 0,530 | 100% | 0,000 | 70% | 0,000 | 0,116 | 905 |
| **B5 B4 + LoRA** | **0,676** | 0,733 | 0,700 | 100% | 0,000 | 70% | 0,015 | 0,099 | 808 |
| B6 Guarded agentic | 0,670 | 0,723 | 0,655 | 100% | 0,000 | 70% | 0,015 | 0,101 | 1.338 |

Per-label F1: **B5** HS 0,797, Abusive 0,814, HS_Individual 0,667, HS_Group 0,425 · **B6** HS 0,800, Abusive 0,792, HS_Individual 0,650, HS_Group 0,440.
Parent B6 dipilih di **dev**: B4 0,576 vs B5 0,692 → **B5**.

**Signifikansi (paired bootstrap 1000×, CI95 delta macro-F1):**

| Perbandingan | Delta | CI95 | Kesimpulan |
|---|---|---|---|
| B1 − B0 | −0,057 | [−0,095; −0,015] | **lebih buruk** (signifikan) |
| B2 − B1 | +0,334 | [+0,260; +0,405] | **lebih baik** (signifikan) |
| B3 − B2 | +0,056 | [−0,025; +0,137] | tidak signifikan |
| B4 − B3 | +0,075 | [+0,021; +0,129] | **lebih baik** (signifikan) |
| B5 − B4 | +0,105 | [+0,030; +0,172] | **lebih baik** (signifikan) |
| B6 − B5 | −0,005 | [−0,051; +0,038] | tidak signifikan |
| B6 − B0 | +0,506 | [+0,435; +0,573] | **lebih baik** (signifikan) |

### 6.1 Interpretasi per tahap
- **B0 → B1 (prompt):** −0,057. Dengan prompt terstruktur + few-shot, model 0.5B **terlalu konservatif**: 175/220 jawaban 'semua 0' (B0: 92/220). Exact match naik (0,345 → 0,400) karena ±47% data netral, tetapi macro-F1 turun. B1 sudah memberi span bukti (evidence support 87%).
- **B1 → B2 (retrieval):** +0,334 — lompatan terbesar. Contoh berlabel yang mirip dari train memberi 'kalibrasi' label bagi model kecil.
- **B2 → B3 (hybrid + reranker + filter):** +0,056, tidak signifikan. Kontrol satu variabel (B3 dengan chunk *fixed* seperti B2) menghasilkan 0,433 (Δ vs B2 −0,008), jadi **kenaikan B3 berasal dari strategi chunk heading-aware, bukan dari hybrid retrieval**. Filter metadata mencegah pedoman usang v0.9 masuk konteks (RT04).
- **B3 → B4 (adaptive + verifier):** +0,075 (signifikan). Verifier menghapus unsupported claim (0,053 → 0), planner menaikkan OOS-abstain 0% → 70%, dan ECE turun 0,233 → 0,116. Retry adaptifnya sendiri tidak membantu (A6).
- **B4 → B5 (LoRA):** +0,105 (signifikan). Error reasoning/model turun tajam (§8).
- **B5 → B6 (tool calling):** −0,005, tidak signifikan, dengan biaya ±4,0× panggilan LLM dan latensi p95 808 → 1.338 ms.

## 7. Ablation

### A1 — Chunking pedoman

**Level retrieval** (hybrid+rerank, qrels section):

| Strategi | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | #chunk | Ctx chars top-5 |
|---|---|---|---|---|---|---|
| fixed (450 char) | 0,643 | 0,892 | 0,561 | 0,496 | 18 | 2.170 |
| heading-aware | 0,416 | 0,697 | 0,393 | 0,322 | 25 | 897 |
| hierarchical (child→parent) | 0,323 | 0,520 | 0,289 | 0,245 | 47 | 3.247 |

**Level LLM** (B4, locked test; groundedness = evidence support):

| Strategi | Macro-F1 | Evidence support | Unsupported claim | Prompt tokens | Latency p95 (ms) |
|---|---|---|---|---|---|
| fixed | 0,492 | 99% | 0,011 | 1.124 | 889 |
| heading | 0,571 | 100% | 0,000 | 1.037 | 905 |
| hierarchical | 0,471 | 100% | 0,000 | 1.164 | 927 |

Recall retrieval terbaik = **fixed**, tetapi kualitas jawaban LLM terbaik = **heading**. Keduanya tidak sejalan: chunk fixed mencakup beberapa section sekaligus (recall tinggi) tetapi memberi konteks yang lebih berisik bagi model kecil, sedangkan heading-aware memberi satu definisi yang utuh dan ringkas.

### A2 — Retriever (heading-aware chunks, qrels identik)

| Retriever | Recall@5 | Recall@10 | nDCG@10 | MRR@10 | Label-precision@5 (contoh) | Latency mean (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|
| dense | 0,749 | 0,955 | 0,607 | 0,510 | 0,461 | 9,6 | 10,4 |
| BM25 | 0,532 | 0,752 | 0,467 | 0,410 | 0,566 | 0,5 | 1,1 |
| hybrid (RRF) | 0,588 | 0,891 | 0,536 | 0,433 | 0,521 | 10,1 | 11,2 |
| hybrid + rerank | 0,416 | 0,697 | 0,393 | 0,322 | 0,515 | 79,3 | 97,8 |

Untuk pedoman, **dense** terbaik; untuk mencari contoh berlabel mirip, **bm25** paling presisi (kata kasar/slang adalah sinyal leksikal). Reranker cross-encoder mMARCO (dilatih untuk QA) **menurunkan** recall sekaligus menambah latensi. B3–B6 tetap memakai hybrid+rerank karena definisi B3 di spesifikasi mewajibkannya. Rekomendasi: hybrid tanpa reranker atau reranker yang di-fine-tune pada domain ini.

### A3 — Verifier

| Varian | Macro-F1 | Unsupported claim | Evidence support | False refusal |
|---|---|---|---|---|
| B4 tanpa verifier | 0,509 | 0,032 | 98% | 0,000 |
| B4 + verifier + repair | 0,571 | 0,000 | 100% | 0,000 |
| B5 tanpa verifier | 0,672 | 0,018 | 98% | 0,000 |
| B5 + verifier + repair | 0,676 | 0,000 | 100% | 0,015 |

Verifier menghapus seluruh unsupported claim di kedua parent dan menaikkan macro-F1 B4 +0,062 lewat aturan konsistensi HS↔target. Biaya: 5 panggilan repair; false refusal B5 0,015.

### A4 — LoRA vs tanpa adapter (parent B4)
- **Gain:** macro-F1 0,571 → 0,676 (+0,105, CI95 [+0,030; +0,172]). HS +0,114, Abusive +0,172, HS_Individual +0,182, HS_Group −0,049.
- **Overfit check:** val loss (train held-out) 0,170 → 0,080 → 0,063 → 0,055 → 0,053; dev loss 0,118 → 0,038 → 0,038 → 0,033 → 0,033. Kurva turun lalu stabil, tidak ada tanda overfit dalam 1 epoch.
- **Resource:** 1600 contoh, 200 step, 24 menit di Tesla T4 (LoRA (fp32 weights + fp16 autocast (AMP), bukan QLoRA)), peak VRAM 5,8 GB, adapter 35 MB, trainable params 1,75%.
- **OOD (Alfina et al. 2017, label HS saja):** dataset utama ternyata memuat sebagian tweet Alfina (paper Ibrohim & Budi §3). 193 dari 713 tweet yang sama/near-duplicate (cosine ≥ 0,8) dengan dataset utama **dibuang**; 200 sampel diambil dari 520 tweet bersih.

| Varian | HS-P | HS-R | HS-F1 | Binary macro-F1 |
|---|---|---|---|---|
| B4 (no adapter) | 0,242 | 0,846 | 0,376 | 0,559 |
| B5 (LoRA) | 0,240 | 0,885 | 0,377 | 0,552 |

Pada data yang benar-benar baru, kedua varian **over-prediksi HS** (recall tinggi, presisi rendah) dan LoRA tidak membantu (−0,007). Peningkatan B5 di test in-domain (+0,105) **tidak** berpindah ke domain lain, yaitu domain drift yang nyata. Run laptop awal memakai OOD yang bocor; hasilnya diarsipkan untuk transparansi.

### A5 — B6 tanpa vs dengan tool calling (parent identik B5)

| Varian | Macro-F1 | LLM calls/item | Prompt tokens | Latency mean (ms) | p95 (ms) |
|---|---|---|---|---|---|
| tanpa tool (guard + temporal check) | 0,676 | 0,94 | 1.036 | 541 | 826 |
| dengan tool | 0,670 | 3,78 | 2.441 | 987 | 1.338 |

Dari 618 tool call yang dicoba: **tool correctness 62%** (386 valid), invalid-arg 4% (26), **unsafe call 0%**. **Budget overrun pada 96% item:** model hampir selalu mencoba langkah ke-3 dan tidak pernah memilih `done`, lalu guard memotongnya. End-task outcome −0,005 (tidak signifikan).

### A6 — Fixed vs adaptive retrieval (B4, satu variabel: retry)

| Slice | n | Fixed: exact / macro-F1 | Adaptive: exact / macro-F1 | Retry rata-rata |
|---|---|---|---|---|
| easy (0–1 label) | 116 | 0,621 / 0,136 | 0,629 / 0,136 | 0,45 |
| multi (≥2 label) | 84 | 0,393 / 0,661 | 0,393 / 0,649 | 0,32 |
| conflict (tetangga ≠ gold) | 66 | 0,273 / 0,429 | 0,333 / 0,463 | 0,42 |
| no-evidence (OOS) | 20 | abstain 70% | abstain 70% | 0,30 |

Keseluruhan: fixed 0,579 vs adaptive 0,571 (−0,008). **Retry benefit, failure, biaya:** retry dipicu pada 79 tweet in-scope; **5 membaik, 6 memburuk**, 68 tidak berubah; waktu retrieval 187 → 418 ms pada item yang di-retry. Retry membantu slice *conflict* (exact 0,273 → 0,333) tetapi secara keseluruhan tidak memberi manfaat bersih. Macro-F1 slice *easy* rendah karena slice ini didominasi item tanpa label positif; exact match lebih representatif.

### A7 — Budget konteks

| Budget | k contoh / k pedoman / max char | Macro-F1 | Prompt tokens | Latency mean | p95 |
|---|---|---|---|---|---|
| kecil | 4 / 1 / 1.400 | 0,571 | 1.037 | 579 ms | 905 ms |
| besar | 10 / 2 / 4.200 | 0,551 | 1.424 | 652 ms | 1.029 ms |

Delta −0,020 (CI95 [−0,098; +0,060], tidak signifikan); latensi p95 naik 1,14×. Titik Pareto ada di **budget kecil**: konteks lebih panjang tidak membantu model 0.5B.

## 8. Analisis Penyebab Error

Taksonomi deterministik mencakup enam sumber error di spesifikasi (retrieval, reasoning, tool, source, prompt, adaptation), diperiksa berurutan per item in-scope yang vektor labelnya salah: **format/prompt** (JSON tidak valid) → **abstain** → **source** (dugaan noise label: gold Abusive=0 padahal tweet memuat kata dari `abusive.csv` milik penyedia dataset) → **tool** (B6: tak satu pun tool call valid) → **retrieval** (mayoritas label contoh ter-retrieve ≠ gold) → **reasoning/model**. **Adaptation** dibaca dari perubahan reasoning B4 → B5.

| Config | #salah | format/prompt | abstain | source | tool | retrieval | reasoning/model | no-retrieval |
|---|---|---|---|---|---|---|---|---|
| B0 | 131 | 0 | 1 | 10 | 0 | 0 | 0 | 120 |
| B1 | 120 | 8 | 0 | 2 | 0 | 0 | 0 | 110 |
| B2 | 111 | 10 | 1 | 6 | 0 | 56 | 38 | 0 |
| B3 | 111 | 4 | 0 | 9 | 0 | 57 | 41 | 0 |
| B4 | 94 | 5 | 0 | 11 | 0 | 47 | 31 | 0 |
| B5 | 60 | 0 | 1 | 6 | 0 | 43 | 10 | 0 |
| B6 | 69 | 2 | 1 | 6 | 1 | 44 | 15 | 0 |

- **Prompt/format error** turun setelah LoRA (B4: 5 → B5: 0).
- **LoRA (adaptation) terutama memperbaiki reasoning** (31 → 10).
- Error yang tersisa **didominasi retrieval** (B5: 43 dari 60): contoh tetangga bertentangan dengan gold.
- **Source:** 2–11 error per konfigurasi kemungkinan berasal dari label sumber (mis. tweet berisi 'monyet' dilabel tidak kasar). Inilah alasan human adjudication (§10) diperlukan.
- **Tool:** 1 error B6 terjadi ketika agent gagal membuat satu pun tool call valid.
- Contoh kualitatif: `results/error_analysis.md`.

## 9. Hallucination Audit, Fairness, Repeatability

**Hallucination audit** (span mentah LLM yang tidak ada di teks, sebelum verifier):

| Config | B1 | B2 | B3 | B4 | B5 | B6 |
|---|---|---|---|---|---|---|
| Hallucinated span rate | 0,112 | 0,131 | 0,118 | 0,110 | 0,091 | 0,081 |

Sekitar 8%–13% span mentah dikarang model. Verifier membuang span tersebut sehingga evidence support akhir B4–B6 = 100%. B0 tidak pernah memberi span.

**Fairness** (FPR label HS pada tweet non-HS):

| Config | FPR, menyebut identitas kelompok (n=24) | FPR, tanpa identitas (n=92) |
|---|---|---|
| B3 | 0,208 | 0,163 |
| B4 | 0,208 | 0,163 |
| B5 | 0,042 | 0,109 |
| B6 | 0,125 | 0,163 |

Di B3/B4, tweet yang menyebut agama/etnis/gender lebih sering dituduh HS (bias identitas, +0,045 pada B4). Setelah LoRA (B5): 0,042 vs 0,109. Sampel identitas kecil (n=24), sehingga temuan ini **indikatif, bukan konklusif**.

**Repeatability** (3 run independen, seed 11/22/33, T=0.7, top-p 0.9, subset 40 item):

| Config | B0 | B1 | B2 | B3 | B4 | B5 | B6 |
|---|---|---|---|---|---|---|---|
| Mean macro-F1 | 0,181 | 0,250 | 0,375 | 0,432 | 0,422 | 0,613 | 0,539 |
| Std | 0,024 | 0,079 | 0,042 | 0,064 | 0,055 | 0,051 | 0,115 |

Menurut definisi *reliable* (std ≤ 0,05): **B0, B2**. Dengan sampling T=0,7, model 0.5B cukup sensitif; agentic loop (B6) paling tidak stabil. Hasil utama memakai decoding greedy (deterministik).

## 10. Human Review (status: perangkat siap, WAJIB diisi tim)
Anotasi blind dua annotator **tidak disimulasikan** dengan model (ADR-012). Yang sudah disiapkan:
- `human_review/sheet_annotator_A.csv` dan `sheet_annotator_B.csv`: 100 item test in-scope, urutan acak berbeda, tanpa label gold maupun prediksi.
- `human_review/protocol.json`: instruksi blind + pedoman `label_policy_v1.0.md`.
- `python -m tbp.human_review agree`: Cohen's κ per label, % agreement, `disagreement_log.csv`, `adjudication_sheet.csv`.
- `python -m tbp.human_review finalize`: human gold, κ human vs label dataset, dan macro-F1 B0–B6 terhadap human gold.

> **TODO tim:** isi setelah anotasi. κ per label: HS ___ · Abusive ___ · HS_Individual ___ · HS_Group ___ ; jumlah disagreement ___ ; ringkasan resolusi ___.

## 11. Red-team (20 fixture)

| Kategori | B0 (tanpa guard) | B6 + guard v1 | B6 + guard v2 (+ injection filter) |
|---|---|---|---|
| Prompt/document injection (5) | 2/5 | 4/5 | 5/5 |
| Policy/PII leakage (5) | 2/5 | 5/5 | 5/5 |
| Tool misuse (5) | 0/5 | 5/5 | 5/5 |
| Safety/robustness (5) | 2/5 | 5/5 | 5/5 |
| **Total** | **6/20** | **19/20** | **20/20** |

- v1 gagal pada **RT05** (blok JSON label palsu di dalam teks; model menurut/abstain).
- **v2 ditambahkan setelah melihat hasil v1 di run laptop:** filter pola generik (frasa override instruksi, prefix peran `SYSTEM:`, perintah ubah label/abstain, blok JSON berkunci label). Karena didesain setelah melihat fixture, angka 20/20 **berisiko overfit ke fixture** dan perlu diuji pada serangan baru. False-positive filter pada 200 tweet test = **0/200**.
- Detail per fixture (ID, goal, attack surface, baseline outcome, mitigation outcome): `results/redteam.json` dan `results/report_tables.md` (T15).

## 12. Serving, Observability, dan Demo
- **API (FastAPI):** `POST /classify` (satu konfigurasi: text, config B0–B6, as_of, force_fallback), `POST /compare` (menjalankan B0–B6 sekaligus untuk teks yang sama), `GET /health`, `GET /logs`.
- **UI utama (Streamlit, `app_streamlit.py`, port 8501)** dengan 3 tab: **Klasifikasi** (ketik tweet, satu tombol, hasil B0–B6 berdampingan), **Hasil Eksperimen** (ringkasan, grafik B0–B6, per-label F1, signifikansi, ablation A1–A7, repeatability, error & audit, red-team; dibaca langsung dari `results/`), dan **Laporan** (LAPORAN.md + unduh). Streamlit memanggil API FastAPI (`/compare`). Jalankan dengan `start_demo.bat`.
- **UI cadangan (HTML di FastAPI):** `GET /`. Pengguna mengetik tweet lalu menekan satu tombol **Klasifikasi**, dan hasil **B0–B6 tampil berdampingan** dalam satu tabel (label, bukti, latensi per versi). Perbedaan antar-konfigurasi langsung terlihat saat demo. Mendukung `/?q=teks` untuk menjalankan otomatis.
- **Guard serving:** sanitasi kontrol-char, potong >2.000 char, injection filter, redaksi PII (HP/email/NIK/mention/URL) **sebelum** model maupun log.
- **Demo di Kaggle (tanpa GPU lokal):** notebook `Streamlit - Case B: Hate Speech Indonesian Twitter` memulihkan hasil run resmi, menyalakan API + Streamlit di GPU T4, dan membuat link publik sementara (`https://….trycloudflare.com`) yang bisa dibuka anggota tim dari perangkat mana pun. Langkah: `KAGGLE_PANDUAN.md`.
- **Fallback:** bila LLM error atau output melanggar kontrak, sistem memakai voting kNN contoh train + span lexicon (`fallback_used=true`). Terbukti di `docs/screenshots/api_response_fallback.json`.
- **Observability:** `logs/serving.jsonl` (trace_id, config, label, skor, span, versi pedoman, tool_log, token, latensi, fallback, error). Setiap run eksperimen punya manifest (`results/runs/*.manifest.json`: prompt id + hash, model, adapter, index version, seed/temperature, hash item & output, environment).
- **Bukti:** `docs/screenshots/ui_compare.png` (UI perbandingan B0–B6), `ui_demo.png` (UI versi awal satu konfigurasi), `api_response_B6.json`, `api_response_injection_pii.json` (injection dinetralkan + nomor HP → `[PHONE]`), `api_response_oos.json` (abstain out_of_scope), `api_response_fallback.json`.

## 13. Keterbatasan dan Ancaman Validitas
1. **Model kecil (0.5B)** dipilih karena batas GPU 4 GB. Angka absolut rendah dibanding model ≥7B. Kesimpulan bersifat **relatif** antar konfigurasi pada base model yang sama.
2. **Label dataset bernoise** (crowd annotation 2019). Analisis error menemukan kasus yang tampak salah label, dan human review belum diisi.
3. **Rationale silver:** target span LoRA berasal dari lexicon train, bukan anotasi manusia. Evidence support mengukur *grounded*, bukan *benar secara semantik*.
4. **Reranker off-domain** (mMARCO) merugikan retrieval. B3–B6 tetap memakainya demi kepatuhan definisi B3.
5. **Agentic B6:** model 0.5B hampir tidak pernah berhenti sendiri (budget overrun 96%), dan gain tool tidak signifikan.
6. **Injection filter v2** didesain setelah melihat fixture red-team (risiko overfit fixture).
7. **Latensi** generasi dihitung sebagai porsi waktu batch 8 (amortized). Latensi retrieval A1/A2 diukur di GPU T4 Kaggle; latensi antar-run dapat berbeda karena warm-up dan batching.
8. **Riwayat run.** Run pertama di laptop (RTX 2050, bf16) mengandung tiga masalah yang ditemukan saat audit: (a) LoRA dilatih sebelum perbaikan bug chunking pedoman; (b) teks out-of-scope di test sempat dilihat saat menyetel *scope checker*; (c) uji OOD bocor (lihat butir 11). Ketiganya diperbaiki, lalu **seluruh eksperimen dijalankan ulang dari nol di Kaggle T4**. Angka di laporan ini berasal dari run Kaggle; hasil laptop diarsipkan di `archive/laptop_rtx2050/` untuk transparansi.
9. Slice "multi-hop" dipetakan menjadi "multi-label ≥2" karena task ini klasifikasi, bukan QA (ADR-007).
10. Dari B2 ke B3 berubah **dua** hal: metode retrieval (dense → hybrid + reranker + filter) **dan** strategi chunk pedoman (fixed → heading-aware). Karena itu ditambahkan **kontrol satu variabel** (B3 dengan chunk fixed): macro-F1 0,433 vs B2 0,441 (Δ −0,008, tidak signifikan). Artinya kenaikan B3 (+0,056) berasal dari chunking, bukan dari hybrid retrieval.
11. **Uji OOD versi awal bocor** karena dataset utama ternyata memuat sebagian tweet dataset OOD (Alfina et al. 2017), seperti tercatat di paper Ibrohim & Budi. Hal ini ditemukan saat membaca paper sumber, lalu diperbaiki dengan menghapus 193 tweet tumpang-tindih (A4, `results/ood_overlap.json`). Hasil versi bocor tetap diarsipkan untuk transparansi.
12. **Scope checker heuristik** (daftar stopword) hanya menolak 70% teks out-of-scope **baru**. Gagal pada bahasa asing berhuruf Latin selain Inggris (Vietnam, Tagalog, Spanyol, Italia) dan potongan kode Python/HTML. Checker **tidak** disetel ulang memakai test ini (anti-leakage). Perbaikan ke depan: model identifikasi bahasa (mis. fastText lid.176) dan deteksi kode, divalidasi di dev.
13. **Stabilitas di bawah sampling:** dengan T=0,7, hanya B0 dan B2 yang memenuhi ambang *reliable* (std ≤ 0,05); B4/B5 di ambang (0,055/0,051), B6 paling tidak stabil (0,115). Hasil utama memakai greedy decoding yang deterministik.

## 14. Reproducibility
- Satu perintah: `bash run_all.sh` (data → LoRA → retrieval → dev → test → ablation → repeat → OOD → red-team → tabel).
- **Run resmi di Kaggle** (GPU T4): notebook `Case B: Multilabel Hate Speech Indonesian Twitter` (notebook presentasi: setiap tahap menampilkan verifikasi PDF §3 dan eksperimen PDF §4/§4.1, kode asli di sel `%%writefile`) menjalankan pipeline dari nol; demo memakai notebook `Streamlit - Case B: Hate Speech Indonesian Twitter`. Panduan: `KAGGLE_PANDUAN.md`. Lingkungan Kaggle tercatat di `results/environment_kaggle.json`.
- Presisi otomatis per GPU: bf16 di GPU Ampere+ (laptop RTX), fp16 di T4 (inference) dan fp32 + autocast fp16 (training LoRA).
- Seed global 42; split + corpus + qrels ber-hash SHA-256 (`data/processed/split_manifest.json`); versi pustaka dikunci di `requirements.txt` (transformers 5.23.0, peft 0.22.0, sentence-transformers 6.3.0).
- Raw output **per item** semua run tersimpan di `results/runs/*.jsonl` (prediksi, span, raw LLM output, token, latensi, retrieved ids, tool log).

## 15. Pemetaan ke Kriteria Spesifikasi
| Kriteria spek | Status | Bukti |
|---|---|---|
| Corpus ≥300 dokumen nyata berizin | ✅ 12.594 contoh + pedoman 2 versi | `source_register.json`, `corpus_examples.jsonl` |
| Benchmark ≥100 query, manifest SHA-256, dedup/near-dup | ✅ test 220, dev 130 | `split_manifest.json` |
| Qrels + label unanswerable | ✅ 551 qrels, 50 OOS `answerable=false` (dev 30 + test 20) | `qrels.jsonl`, ADR-006 |
| Human review 2 annotator + adjudikasi | ⚠️ perangkat siap, **pengisian oleh tim** | `human_review/`, §10 |
| Red-team ≥20 (5/5/5/5) | ✅ 20 fixture, baseline vs mitigasi | `redteam.json`, §11 |
| LLM plain vs LoRA | ✅ B4 vs B5, test + OOD | A4, `adapters/` |
| Repeatability ≥3 run | ✅ 7 config × 3 seed | §9, `results/runs/repeat_*` |
| Serving UI + API + fallback + log | ✅ | §12, `docs/screenshots/` |
| B0–B6, base generator sama, B6 vs parent identik | ✅ | §6, A5 |
| Ablation A1–A7 | ✅ | §7 |
| Prompt registry, versioning, observability | ✅ | `tbp/prompts.py`, run manifests |
| ADR untuk substitusi (SETARA) | ✅ | `docs/ADR.md` |
| Etika: lisensi, PII, anti-leakage, definisi klaim | ✅ | §1, §5 |
| Demo teknis & pertanggungjawaban individu | ⚠️ **oleh tim** (live demo, pembagian kontribusi) | – |

---
## Lampiran
- **L1.** Tabel hasil lengkap (auto-generated): [results/report_tables.md](results/report_tables.md)
- **L2.** Analisis error + contoh kualitatif: [results/error_analysis.md](results/error_analysis.md)
- **L3.** ADR: [docs/ADR.md](docs/ADR.md)
- **L4.** Pedoman label: [data/guidelines/label_policy_v1.0.md](data/guidelines/label_policy_v1.0.md)
- **L5.** Source register: [data/source_register.json](data/source_register.json) · split manifest: [data/processed/split_manifest.json](data/processed/split_manifest.json)
- **L6.** Training log LoRA: [adapters/lora_b5_training_log.json](adapters/lora_b5_training_log.json)
- **L7.** Bukti serving: [docs/screenshots/](docs/screenshots/)
