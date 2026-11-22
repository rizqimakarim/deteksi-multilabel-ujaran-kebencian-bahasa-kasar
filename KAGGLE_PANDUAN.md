# Panduan Menjalankan Proyek di Kaggle (GPU T4)

Untuk semua anggota Kelompok Rizqi. Tidak perlu laptop dengan GPU — cukup akun Kaggle.
Belum paham alur proyeknya? Baca dulu `ALUR_END_TO_END.md` (urutan kerja dari unduh dataset sampai dashboard).

## Isi di Kaggle (akun pemilik: `rizqimakarim`, semuanya **Private**)
| Nama | Jenis | Fungsi |
|---|---|---|
| `Case B: Multi-label Hate Speech Indonesian Twitter` | **Dataset** | Kode + data mentah + pedoman label + lembar human review |
| `Case B: Multilabel Hate Speech Indonesian Twitter` | **Notebook 1** (eksperimen) | Menjalankan **seluruh eksperimen** dari nol (±2–3 jam, GPU T4) |
| `Streamlit - Case B: Hate Speech Indonesian Twitter` | **Notebook 2** (demo) | Menyalakan **aplikasi Streamlit** + link publik untuk demo |

## Syarat akun
1. **Verifikasi nomor HP** (Kaggle → Settings → Phone verification). Tanpa ini GPU & Internet tidak bisa dinyalakan.
2. Anggota tim lain: minta pemilik menambahkan Anda sebagai **collaborator** di dataset & kedua notebook (tombol *Share*).
   Dataset bersifat **private** karena data OOD (Alfina 2017) dilarang disebarkan publik.

## A. Menjalankan ulang eksperimen (Notebook 1)
1. Buka **notebook** `Case B: Multilabel Hate Speech Indonesian Twitter` → **Edit** (atau *Copy & Edit* bila Anda bukan pemilik).
2. Panel kanan → **Session options**: *Accelerator* = **GPU T4 x2** (cukup), *Internet* = **On**.
3. Pastikan **Input** berisi dataset `Case B: Multi-label Hate Speech Indonesian Twitter`.
4. Klik **Save Version** → **Save & Run All (Commit)**. Boleh menutup browser; proses berjalan di server Kaggle.
5. Setelah selesai (status *Complete*), hasil ada di tab **Output**: folder `tbp_project/` dan file `tbp_output.zip`.

## B. Demo aplikasi (Notebook 2)
1. Buka notebook `Streamlit - Case B: Hate Speech Indonesian Twitter` → **Edit**.
2. *Accelerator* = **GPU T4**, *Internet* = **On**.
3. **Add Input**: dataset `Case B: Multi-label Hate Speech Indonesian Twitter` **dan** output **notebook** `Case B: Multilabel Hate Speech Indonesian Twitter`.
4. Klik **Run All**. Di bawah sel *tunnel* muncul:
   `BUKA DEMO DI: https://xxxx.trycloudflare.com`
5. Buka link itu (bisa dari HP/laptop mana pun). Isinya sama dengan versi laptop:
   **Klasifikasi** (B0–B6 berdampingan) · **Hasil Eksperimen B0–B6** · **Analisis Lain** · **Laporan**.
6. Link aktif selama sesi notebook menyala. Matikan sesi setelah demo agar kuota GPU tidak terbuang.

## C. Presentasi
- **Notebook 1** (`Case B: Multilabel Hate Speech Indonesian Twitter`) adalah notebook presentasi. Buka versi terakhir yang sudah *Complete*, lalu gulir dari atas ke bawah:
  bagian 1–14 mengikuti PDF. Setiap tahap menampilkan **✔ Verifikasi PDF §3** (standar wajib + cara verifikasi + buktinya)
  dan **🧪 Eksperimen PDF §4 / §4.1** (fitur pembeda/hipotesis + eksperimen + hasil).
- Kode asli setiap modul ada di sel `%%writefile` yang tersembunyi. Klik **Show code** untuk menunjukkannya kepada dosen.
- Bagian 14 berisi tabel **kepatuhan PDF** (✅/⚠️) yang dihitung otomatis.
- **Notebook 2** (`Streamlit - Case B: …`) untuk demo langsung: ketik tweet → hasil B0–B6 berdampingan.

## Kuota & waktu
- GPU gratis ±30 jam/minggu per akun; satu sesi maksimal 12 jam.
- Notebook 1: ±2–3 jam. Notebook 2: sebatas lama demo.

## Menjalankan di laptop (opsional)
- Dengan GPU NVIDIA: lihat `README.md` (`bash run_all.sh`, lalu `start_demo.bat`).
- **Tanpa GPU** pun aplikasi tetap jalan (lebih lambat, ±30 detik per versi): hasil eksperimen & laporan tidak butuh GPU.

> Catatan: dataset bernama `Case B: Multi-label …` (menu **Datasets**), Notebook 1 bernama `Case B: Multilabel …` (menu **Code**). Kaggle tidak mengizinkan notebook dan dataset memakai alamat yang persis sama.
