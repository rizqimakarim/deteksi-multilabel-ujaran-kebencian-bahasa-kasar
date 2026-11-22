# Alur End-to-End Proyek TBP — Kelompok Rizqi

Dokumen ini menceritakan **urutan pekerjaan dari awal sampai akhir**: dari mengunduh dataset mentah sampai aplikasi (dashboard) Streamlit dan notebook presentasi di Kaggle.
Setiap langkah mencatat **apa yang dikerjakan**, **berkas masuk → berkas keluar**, dan **perintahnya**.
Angka hasil lengkap ada di `RINGKASAN_PROYEK.md` dan `LAPORAN.md`.

```
 [1] Unduh dataset ──► [2] Tulis pedoman label ──► [3] Persiapan data ──► [4] Latih LoRA
                                                                              │
 [9] Pindah ke Kaggle ◄── [8] Aplikasi ◄── [7] Laporan ◄── [6] Eksperimen ◄── [5] Bangun B0–B6
        │
        ▼
 [10] Presentasi: Notebook 1 (alur + bukti) + Notebook 2 (demo Streamlit)
```

---

## 0. Menentukan kasus
- Spesifikasi tugas: `NLP_IR.pdf` (*TBP: Frontier LLM Research & Engineering*). File ini tidak ada di GitHub karena materi dosen; salinannya ada di dataset Kaggle (private).
- Dipilih **Case B: klasifikasi teks**. Atas izin dosen, versinya **multi-label**: satu tweet bisa punya beberapa label sekaligus.
- Judul: **Evidence-Aware Multi-label Hate Speech and Abusive Language Detection pada Twitter Indonesia**.
- Input: 1 tweet berbahasa Indonesia. Output: 4 label (`HS`, `Abusive`, `HS_Individual`, `HS_Group`) + bukti kata + versi pedoman + abstain bila teks di luar cakupan.

## 1. Mengunduh dataset
| Data | Asal | Disimpan di |
|---|---|---|
| Dataset utama: 13.169 tweet, 12 kolom label | GitHub Ibrohim & Budi (2019): https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection | `data/raw/re_dataset.csv` |
| Kamus slang → kata baku | repo yang sama | `data/raw/new_kamusalay.csv` |
| Daftar kata kasar | repo yang sama | `data/raw/abusive.csv` |
| Dataset uji luar domain (OOD): 713 tweet | GitHub Alfina et al. (2017): https://github.com/ialfina/id-hatespeech-detection | `data/ood/IDHSD_RIO_unbalanced_713_2017.txt` |
| Paper sumber (untuk sitasi) | ACL Anthology W19-3506 dan ICACSIS 2017 | `docs/references/` |

- Tweet **tidak diambil langsung dari Twitter/X**. Kita memakai dataset yang sudah dikumpulkan dan dilabeli peneliti; username dan URL sudah diganti `USER` dan `URL`.
- Dari 12 kolom label, dipakai 4: `HS`, `Abusive`, `HS_Individual`, `HS_Group`.
- Lisensi dataset utama CC BY-NC-SA 4.0, wajib sitasi. Dataset Alfina **tidak boleh disebarkan ulang**, karena itu dataset Kaggle kita dibuat **private**.

## 2. Menulis pedoman label (dokumen untuk retrieval)
- Tim menulis dua versi pedoman:
  - `data/guidelines/label_policy_v1.0.md`: versi berlaku mulai 2026-07-01, berisi definisi label, contoh, kasus ambigu, kebijakan abstain, dan aturan bukti.
  - `data/guidelines/label_policy_v0.9.md`: versi lama yang sudah usang. Dipakai untuk menguji apakah sistem memilih versi yang berlaku pada tanggal tertentu (*temporal check*).
- Pedoman ini menjadi "buku aturan" yang dicari dan dikutip oleh RAG (B2–B6).

## 3. Persiapan data
**Perintah:** `python -m tbp.data_prep` · **Kode:** `tbp/data_prep.py`, `tbp/textutil.py`
**Masuk:** `data/raw/*`, `data/guidelines/*` → **Keluar:** `data/processed/*`

1. **Cleaning:** membuang `RT`, mention berulang, kode emoji mentah, dan spasi berlebih, lalu menormalkan kata slang dengan kamus alay.
2. **Dedup:** 265 duplikat dibuang, ditambah 15 teks yang sama tapi labelnya berbeda.
3. **Split terkunci** (stratified, seed 42, dicatat dengan hash SHA-256 di `split_manifest.json`):
   - Train 12.594: corpus retrieval, few-shot, dan training LoRA.
   - Dev 130: 100 tweet + 30 teks luar cakupan. Hanya untuk memilih/menyetel.
   - Test 220: 200 tweet + 20 teks luar cakupan baru. Hanya untuk angka akhir dan tidak pernah dipakai menyetel.
4. **Near-duplicate check:** 10 tweet train yang terlalu mirip dengan dev/test dibuang, sehingga overlap train–test = 0.
5. **Teks luar cakupan (OOS):** kalimat bahasa asing, kode, angka, simbol, emoji, dan teks kosong. Sistem wajib menolak menjawab (*abstain*) untuk teks seperti ini.
6. **Chunking pedoman** dengan 3 cara: `fixed`, `heading`, `hierarchical` (`guide_chunks_*.jsonl`).
7. **Qrels:** 551 penilaian bagian pedoman mana yang relevan untuk tiap tweet dev/test (`qrels.jsonl`). Dipakai untuk mengukur retrieval.
8. **Embedding** contoh dan pedoman memakai multilingual-e5-small (`emb_*.npy`), plus `lexicon.json` dan `data/source_register.json` (daftar sumber dan lisensi).

## 4. Melatih adapter LoRA
**Perintah:** `python -m tbp.train_lora` · **Kode:** `tbp/train_lora.py`
**Masuk:** `data/processed/train.jsonl` → **Keluar:** `adapters/lora_b5/`, `adapters/lora_b5_training_log.json`

- Model dasar **Qwen2.5-0.5B-Instruct** (open-weight, Apache-2.0) dilatih dengan **LoRA r=16** pada 1.600 tweet train.
- Lama latih ±24 menit di GPU T4 (fp16 + AMP). Ukuran adapter ±35 MB; model aslinya tidak diubah.
- Adapter ini dipakai oleh B5 dan B6.

## 5. Membangun 7 konfigurasi B0–B6
**Kode:** `tbp/prompts.py`, `tbp/retrieval.py`, `tbp/pipelines.py`, `tbp/agent.py`, `tbp/llm.py`

Ketujuh konfigurasi memakai model dasar yang sama. Setiap tingkat menambah **satu** kemampuan, supaya efek tiap tambahan bisa diukur:

| ID | Yang ditambahkan |
|---|---|
| B0 | LLM langsung (zero-shot) |
| B1 | Prompt terstruktur: definisi, aturan, 4 few-shot, kebijakan abstain |
| B2 | + RAG sederhana: dense retrieval dan chunk fixed |
| B3 | + RAG hybrid: BM25 + dense + fusion RRF + reranker + filter versi pedoman |
| B4 | + query planner, skor kecukupan bukti, retry, verifier bukti, abstain |
| B5 | + adapter LoRA (langkah 4) |
| B6 | + agent yang memanggil tool sendiri, dengan guard (otorisasi, validasi argumen, cek tanggal, batas langkah) |

## 6. Menjalankan eksperimen
**Perintah:** `bash run_all.sh` (menjalankan langkah 3, 4, 6, dan 7 berurutan) · **Kode:** `tbp/run_experiments.py`, `tbp/evaluate.py`, `tbp/redteam.py`
**Keluar:** `results/results.json`, `results/slices.json`, `results/redteam.json`, `results/runs/`, `logs/*.log`

| Tahap | Isi |
|---|---|
| `retrieval` | A1 (3 cara chunking) dan A2 (BM25 / dense / hybrid / rerank), diukur dengan Recall@k, MRR, nDCG terhadap qrels |
| `dev` | Memilih setelan di data dev (test tetap terkunci) |
| `test` | B0–B6 pada 220 data test: macro/micro/per-label F1, exact match, hamming, bukti valid, klaim tanpa bukti, abstain, ECE, biaya, latensi, plus uji signifikansi bootstrap |
| `ablations` | A3 verifier, A4 LoRA, A5 tool, A6 retrieval adaptif, A7 budget konteks, kontrol satu variabel B3 |
| `repeat` | 7 konfigurasi × 3 seed (T=0,7) untuk melihat kestabilan |
| `ood` | Uji pada tweet Alfina 2017. Sebelumnya, 193 tweet yang mirip dataset utama dibuang supaya tidak bocor |
| red-team | 20 serangan: prompt injection, kebocoran PII, penyalahgunaan tool, robustness. Dibandingkan antara B0, B6 tanpa filter, dan B6 dengan filter |

## 7. Membuat laporan
**Perintah:** `python -m tbp.make_report` lalu `python tools/update_laporan.py`
**Keluar:** `results/report_tables.md`, `results/error_analysis.md`, bagian hasil di `LAPORAN.md`

- Tabel dan grafik dibuat otomatis dari `results/results.json`, jadi angka di laporan selalu sama dengan hasil run.
- Analisis error mengelompokkan penyebab salah: format, abstain, retrieval, penalaran, tool, dan sumber.
- Bagian ini juga mencakup audit halusinasi (bukti yang dikarang model) dan cek fairness (tweet netral yang menyebut agama/etnis/gender).
- Alasan setiap keputusan desain dicatat di `docs/ADR.md`.

## 8. Membuat aplikasi (dashboard)
**Kode:** `tbp/serve.py`, `tbp/serving_guard.py`, `app_streamlit.py` · **Jalankan di laptop:** dobel-klik `start_demo.bat`

- **API FastAPI** (port 8000):
  - `POST /classify`: satu konfigurasi;
  - `POST /compare`: B0–B6 sekaligus;
  - `GET /health` dan `GET /logs`.
- **Pengaman:** nomor HP, email, dan NIK disensor sebelum masuk model dan log. Teks yang terlalu panjang dipotong dan prompt injection dinetralkan. Kalau LLM gagal, ada fallback kNN.
- **Dashboard Streamlit** (port 8501), 4 tab:
  1. 🔎 **Klasifikasi**: ketik tweet, klik sekali, hasil B0–B6 tampil berdampingan (label, bukti, skor, abstain).
  2. 🧪 **Hasil Eksperimen B0–B6**: 7 sub-tab, judulnya sesuai kolom *Eksperimen* di PDF.
  3. 📈 **Analisis Lain**: per-label, signifikansi, ablation A1–A7, repeatability, error & audit, red-team.
  4. 📄 **Laporan**: isi `LAPORAN.md`, sumber/sitasi dataset, dan daftar sumber.

## 9. Pindah ke Kaggle (GPU T4) — run resmi
**Kode:** `tools/make_kaggle_package.py`, `tools/story_notebook.py` · **Panduan:** `KAGGLE_PANDUAN.md`

1. `python tools/make_kaggle_package.py rizqimakarim` membungkus kode, data mentah, pedoman, dan dokumen menjadi `dist/kaggle_dataset/tbp_project.zip`, serta membuat 2 notebook.
2. Zip tersebut diupload sebagai **Dataset** `Case B: Multi-label Hate Speech Indonesian Twitter` (private).
3. **Notebook 1** `Case B: Multilabel Hate Speech Indonesian Twitter` menjalankan langkah 3–7 dari nol di GPU T4 (±76 menit). Hasilnya inilah **angka resmi**.
   - Dua kali run memberi angka kualitas yang identik dan adapter yang sama persis sampai byte, jadi hasilnya bisa diulang.
   - Hasil run lama di laptop disimpan di `archive/laptop_rtx2050/`, tetapi bukan angka resmi.
4. Output Notebook 1 (`tbp_output.zip`) diunduh ke laptop: `results/`, `adapters/`, `data/processed/`, `logs/`. Laporan lalu diperbarui dari output itu.
5. **Notebook 2** `Streamlit - Case B: Hate Speech Indonesian Twitter` memakai output Notebook 1, menyalakan API + Streamlit, lalu membuat link publik `https://xxxx.trycloudflare.com` lewat Cloudflare tunnel.

## 10. Presentasi
- **Notebook 1** dibaca dari atas ke bawah. Bagian 1–14 mengikuti urutan PDF. Setiap tahap menampilkan:
  - **✔ Verifikasi PDF §3**: standar wajib, cara verifikasi, dan buktinya;
  - **🧪 Eksperimen PDF §4 / §4.1**: hipotesis, eksperimen, dan hasilnya.

  Kode asli setiap modul ada di sel tersembunyi (*Show code*). Bagian 14 berisi tabel kepatuhan PDF (✅/⚠️).
- **Notebook 2**: Run All, buka link tunnel, ketik tweet, dan hasil B0–B6 keluar berdampingan.

## 11. Yang masih dikerjakan tim
1. **Human review**: dua anggota mengisi `human_review/sheet_annotator_A.csv` dan `sheet_annotator_B.csv` secara terpisah, lalu jalankan `python -m tbp.human_review agree` dan `finalize`. Setelah itu upload ulang dataset Kaggle.
2. **Nama, NIM, dan kontribusi** anggota di bagian atas `LAPORAN.md`.
3. **Demo langsung** saat presentasi.
