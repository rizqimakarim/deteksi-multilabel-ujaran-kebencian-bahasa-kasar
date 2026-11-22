# Ringkasan Proyek TBP — Kelompok Rizqi

## 1. Judul
**Evidence-Aware Multi-label Hate Speech and Abusive Language Detection pada Twitter Indonesia**
(Use Case B — varian multi-label sesuai arahan dosen)

Sistem menerima satu tweet berbahasa Indonesia informal/campur bahasa dan menghasilkan:
- **4 label biner** (boleh lebih dari satu bernilai 1):

  | Label | Arti |
  |---|---|
  | `HS` | Ujaran kebencian (menyerang/merendahkan pihak tertentu) |
  | `Abusive` | Mengandung kata kasar/makian/cabul |
  | `HS_Individual` | Ujaran kebencian yang targetnya satu orang |
  | `HS_Group` | Ujaran kebencian yang targetnya kelompok (agama, etnis, partai, dll.) |

- **Bukti (evidence span)**: potongan kata dari tweet yang menjadi alasan label.
- **Skor per label**, **versi pedoman label** yang dipakai, dan **abstain** (menolak menjawab) bila teks di luar cakupan.

Contoh: *"USER dasar goblok lu, cebong semua sama aja"* → HS=1, Abusive=1, HS_Individual=1, HS_Group=0, bukti: `goblok`, `cebong`.

Alasan memilih multi-label: setiap label adalah keputusan ya/tidak yang independen, jadi tidak membingungkan di level teks. Ini berbeda dengan sentimen positif-negatif yang bisa muncul bersamaan dalam satu kalimat.

---

## 2. Dataset

### 2.1 Dataset utama
| Item | Keterangan |
|---|---|
| Nama | Indonesian Multi-label Hate Speech and Abusive Language Dataset |
| Sumber | Ibrohim & Budi (2019), ALW3 @ ACL — https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection |
| Isi | 13.169 tweet Indonesia, 12 kolom label (dipakai 4 label) |
| Lisensi | CC BY-NC-SA 4.0 (boleh untuk akademik, wajib sitasi) |
| Privasi | Username dan URL sudah dianonimkan menjadi `USER` dan `URL` oleh penyedia |
| Pendukung | `new_kamusalay.csv` (kamus slang → baku) dan `abusive.csv` (daftar kata kasar), dari repo yang sama |

### 2.2 Dataset OOD (uji luar domain)
| Item | Keterangan |
|---|---|
| Nama | ID Hate Speech Detection (Alfina et al., 2017) — https://github.com/ialfina/id-hatespeech-detection |
| Isi | 713 tweet, label HS / Non_HS; dipakai 200 sampel |
| Lisensi | Bebas dipakai dengan sitasi, **dilarang disebarkan ulang** (tidak boleh di-upload ke repo publik) |

### 2.3 Dokumen pedoman label (dibuat tim)
- `label_policy_v1.0.md` — berlaku mulai 2026-07-01 (definisi label, contoh, kasus ambigu, kebijakan abstain, aturan bukti).
- `label_policy_v0.9.md` — versi lama/usang (berlaku 2025-01-01 s/d 2026-06-30), dipakai untuk menguji *temporal checking*.

---

## 3. Persiapan Data
1. **Cleaning**: hapus kode emoji mentah, `RT`, mention berulang, spasi berlebih.
2. **Dedup**: 265 duplikat dibuang, 15 teks yang sama tapi labelnya berbeda dibuang.
3. **Split terkunci** (dengan hash SHA-256 di `split_manifest.json`):

   | Split | Jumlah | Keterangan |
   |---|---|---|
   | Train | 12.594 | untuk corpus retrieval, few-shot, dan training LoRA |
   | Dev | 130 | 100 tweet + 30 teks di luar cakupan; untuk memilih konfigurasi & menyetel pendeteksi bahasa |
   | Test (locked) | 220 | 200 tweet + 20 teks di luar cakupan; hanya untuk evaluasi akhir |

4. **Near-duplicate check**: 10 tweet train yang terlalu mirip dengan dev/test dibuang. Overlap train vs test = 0.
5. **Teks di luar cakupan** (ditulis tim): kalimat Inggris, bahasa lain, kode program, angka, simbol, emoji, teks kosong → wajib abstain. 30 teks untuk dev (dipakai menyetel pendeteksi bahasa) dan **20 teks baru** untuk test (tidak pernah dipakai menyetel apa pun).
6. **Corpus retrieval**: 12.594 tweet train berlabel + pedoman label (dipotong dengan 3 cara: fixed, heading-aware, hierarchical).
7. **Qrels**: 551 penilaian relevansi (bagian pedoman mana yang relevan untuk tiap tweet dev/test).

---

## 4. Model
| Peran | Model |
|---|---|
| LLM generator (sama untuk B0–B6) | Qwen2.5-0.5B-Instruct (open-weight, Apache-2.0) |
| Dense retrieval | multilingual-e5-small |
| Sparse retrieval | BM25 (dengan normalisasi kata slang) |
| Reranker | mmarco-mMiniLMv2 cross-encoder |
| Fine-tuning | LoRA r=16 pada 1.600 tweet train, 24 menit di Kaggle GPU T4 |

Semua model open-weight dijalankan sendiri (Kaggle GPU T4 untuk run resmi, atau laptop), jadi tidak ada data yang dikirim ke layanan AI pihak luar.

---

## 5. Tujuh Konfigurasi yang Dibandingkan (B0–B6)
| ID | Isi |
|---|---|
| B0 | LLM langsung, zero-shot |
| B1 | Prompt terstruktur: definisi label, aturan, 4 contoh few-shot, kebijakan abstain |
| B2 | B1 + RAG sederhana (dense retrieval, chunk tetap) |
| B3 | RAG hybrid: BM25 + dense + fusion + reranker + filter metadata/versi pedoman |
| B4 | B3 + query planner, skor kecukupan bukti, retry retrieval, verifier bukti, abstain |
| B5 | B4 + adapter LoRA |
| B6 | B5 + agent yang memanggil tool sendiri, dengan guard (otorisasi, validasi argumen, cek tanggal, batas langkah) |

---

## 6. Hasil Utama (test terkunci, run resmi Kaggle GPU T4)
| Config | Macro-F1 | Bukti valid | Abstain teks luar cakupan |
|---|---|---|---|
| B0 | 0,164 | 0% | 0% |
| B1 | 0,107 | 87% | 0% |
| B2 | 0,441 | 92% | 0% |
| B3 | 0,496 | 96% | 0% |
| B4 | 0,571 | 100% | 70% |
| **B5** | **0,676** | 100% | 70% |
| B6 | 0,670 | 100% | 70% |

**Kenaikan yang signifikan secara statistik** (bootstrap 1000×):
- RAG (B1 → B2): +0,334
- RAG adaptif + verifier (B3 → B4): +0,075
- LoRA (B4 → B5): +0,105
- Total B0 → B6: +0,506

**Tidak signifikan**: hybrid vs naive RAG (B2 → B3, +0,056) dan tool calling (B5 → B6, −0,005).

**Temuan penting lain:**
- Prompt terstruktur saja (B1) membuat model 0.5B terlalu hati-hati: 175 dari 220 jawaban "semua 0".
- Kenaikan B2 → B3 ternyata berasal dari **cara memotong pedoman**, bukan dari hybrid retrieval (dibuktikan dengan kontrol satu variabel).
- Reranker justru menurunkan kualitas retrieval; dense retrieval paling baik untuk pedoman.
- LoRA terutama memperbaiki kesalahan penalaran model (31 → 10 kasus). Sisa error kebanyakan dari contoh retrieval yang menyesatkan, dan sebagian tampak sebagai label dataset yang keliru.
- Pendeteksi bahasa hanya menolak 70% teks jebakan baru: lolos untuk bahasa asing berhuruf Latin selain Inggris dan potongan kode.

---

## 7. Ablation (A1–A7)
| Kode | Yang dibandingkan | Hasil singkat |
|---|---|---|
| A1 | Cara memotong pedoman | Untuk pencarian: fixed terbaik (Recall@10 0,89). Untuk jawaban AI: heading-aware terbaik (0,571 vs 0,492) |
| A2 | Jenis retriever | Dense paling baik untuk pedoman, BM25 untuk contoh; reranker menurunkan skor |
| A3 | Tanpa vs dengan verifier | Klaim tanpa bukti dihapus (B4: 3,2% → 0%), macro-F1 B4 +0,062 |
| A4 | Tanpa vs dengan LoRA | +0,105 di test; tidak overfit; di data OOD bersih tidak membantu (0,559 → 0,552) |
| A5 | B6 tanpa vs dengan tool | −0,005 (tidak signifikan); 0% panggilan tool berbahaya; biaya ±4× panggilan LLM |
| A6 | Retrieval tetap vs adaptif | Retry: 5 tweet membaik, 6 memburuk; hanya membantu kasus bukti bertentangan |
| A7 | Budget konteks kecil vs besar | Budget besar tidak lebih baik (−0,020) dan lebih lambat |

---

## 8. Evaluasi Tambahan
- **Repeatability**: 7 konfigurasi × 3 run dengan seed berbeda (sampling T=0,7). B5 tetap terbaik (rata-rata 0,613); B6 paling tidak stabil (std 0,115).
- **Hallucination audit**: 8–13% bukti mentah dari model dikarang (tidak ada di teks); verifier membuangnya.
- **Fairness**: tanpa LoRA, tweet netral yang menyebut agama/etnis/gender lebih sering salah dituduh HS; setelah LoRA selisihnya hilang (sampel kecil, indikatif).
- **Red-team (20 serangan)**: 5 prompt injection, 5 kebocoran data/PII, 5 penyalahgunaan tool, 5 robustness.

  | Sistem | Lolos |
  |---|---|
  | B0 tanpa guard | 6/20 |
  | B6 + guard | 19/20 |
  | B6 + guard + filter injection | 20/20 (filter dibuat setelah melihat kasus gagal di run awal) |

---

## 9. Aplikasi (Serving)
- **API** (FastAPI): `POST /classify` (satu versi), `POST /compare` (B0–B6 sekaligus), `GET /health`, `GET /logs`.
- **UI (Streamlit, http://localhost:8501)** dengan 4 tab: **Klasifikasi** (ketik tweet, klik, hasil B0–B6 berdampingan), **Hasil Eksperimen B0–B6** (7 sub-tab sesuai kolom *Eksperimen* di PDF), **Analisis Lain** (per-label, signifikansi, ablation, repeatability, error, red-team), **Laporan** (isi LAPORAN.md + sumber/sitasi).
- **Di Kaggle**: Notebook 2 `Streamlit - Case B: …` menyalakan aplikasi yang sama dan memberi link publik (lihat `KAGGLE_PANDUAN.md`).
- **UI web** di http://127.0.0.1:8000 untuk mencoba tweet langsung.
- **Pengaman**: nomor HP/email/NIK disensor sebelum masuk model dan log; teks terlalu panjang dipotong; prompt injection dinetralkan.
- **Fallback**: kalau LLM gagal, sistem tetap menjawab dengan voting contoh mirip.
- **Log**: setiap permintaan dicatat di `logs/serving.jsonl` (label, bukti, versi pedoman, token, latensi).

Cara menjalankan: **dobel-klik `start_demo.bat`**, lalu buka http://localhost:8501

---

## 10. Yang Masih Harus Dikerjakan Tim
1. **Human review**: dua anggota mengisi `human_review/sheet_annotator_A.csv` dan `sheet_annotator_B.csv` secara terpisah, lalu jalankan `python -m tbp.human_review agree` dan `finalize`.
2. **Nama, NIM, dan kontribusi** tiap anggota di bagian atas `LAPORAN.md`.
3. **Demo langsung** di depan dosen.

---

## 11. Lokasi File Penting
| File | Isi |
|---|---|
| `ALUR_END_TO_END.md` | Urutan kerja dari unduh dataset sampai dashboard (baca ini dulu) |
| `LAPORAN.md` | Laporan lengkap |
| `README.md` | Cara instalasi dan menjalankan |
| `docs/ADR.md` | Alasan setiap keputusan desain |
| `results/report_tables.md` | Semua tabel hasil |
| `results/error_analysis.md` | Analisis penyebab error + contoh |
| `results/redteam.json` | Detail 20 kasus red-team |
| `docs/screenshots/` | Screenshot UI dan contoh respons API |
| `run_all.sh` | Menjalankan ulang seluruh eksperimen |
| `KAGGLE_PANDUAN.md` | Cara menjalankan eksperimen & demo Streamlit di Kaggle (GPU T4) |
| `archive/laptop_rtx2050/` | Arsip run awal di laptop (bukan angka resmi) |
