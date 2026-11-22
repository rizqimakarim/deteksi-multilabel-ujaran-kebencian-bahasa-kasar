# Arsip hasil run laptop (RTX 2050, bf16) — 20 Nov 2026

Hasil eksperimen pertama yang dijalankan di laptop (GPU RTX 2050 4 GB, bf16). **Bukan angka resmi.**
Angka resmi laporan diambil dari run ulang di **Kaggle GPU T4** (lihat `results/`), yang juga memuat perbaikan:
- test out-of-scope memakai 20 teks baru yang tidak pernah dipakai menyetel scope checker,
- uji OOD dibersihkan dari tweet yang tumpang-tindih dengan dataset utama,
- LoRA dilatih ulang setelah perbaikan bug chunking pedoman,
- tambahan A1 level LLM, B3 satu variabel, kategori error tool/source.
Arsip ini disimpan untuk transparansi dan perbandingan reproducibility antar-GPU.
