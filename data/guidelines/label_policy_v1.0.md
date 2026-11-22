---
doc_id: label_policy
version: "1.0"
valid_from: "2026-07-01"
valid_to: "9999-12-31"
author: Tim TBP
license: CC BY-NC-SA 4.0 (turunan dari skema label Ibrohim & Budi, 2019)
---

# Pedoman Label Ujaran Kebencian dan Bahasa Kasar (Label Policy) v1.0

Dokumen ini adalah pedoman anotasi dan klasifikasi untuk tweet berbahasa Indonesia informal dan campur bahasa. Satu teks boleh memiliki lebih dari satu label (multi-label). Setiap label diputuskan secara biner: 1 berarti label berlaku, 0 berarti tidak berlaku.

## 1. Ruang Lingkup

### 1.1 Teks yang masuk cakupan
Teks pendek berbahasa Indonesia, bahasa gaul, singkatan alay, atau campuran Indonesia-Inggris dan Indonesia-daerah. Contoh: "lu tuh goblok bgt sih", "dasar cebong kampret", "semoga sukses ya kak". Placeholder USER dan URL adalah hasil anonimisasi dan tidak bermakna.

### 1.2 Teks di luar cakupan
Teks yang sepenuhnya berbahasa asing (Inggris penuh, Jepang, Arab, Spanyol, dan lainnya), teks kosong, hanya angka, hanya simbol atau emoji, hanya URL, atau potongan kode program. Untuk teks di luar cakupan sistem wajib abstain (lihat Bagian 8).

## 2. Label HS (Hate Speech / Ujaran Kebencian)

### 2.1 Definisi HS
HS = 1 bila teks menyerang, merendahkan, menghina, memfitnah, atau menghasut kebencian terhadap individu atau kelompok tertentu. Serangan dapat berupa hinaan, julukan merendahkan, tuduhan tanpa dasar, ajakan memusuhi, atau ancaman. Contoh pola: "dasar antek asing pengkhianat", "kafir laknat", "presiden boneka bodoh", "cebong/kampret/kadrun sampah".

### 2.2 Aturan konsistensi HS
Jika HS = 1 maka tepat satu dari HS_Individual atau HS_Group harus bernilai 1. Jika HS = 0 maka HS_Individual = 0 dan HS_Group = 0. HS_Individual dan HS_Group tidak boleh bernilai 1 bersamaan.

### 2.3 Yang bukan HS
Kritik kebijakan yang sopan, berita netral, pendapat tanpa merendahkan, atau kata kasar yang tidak ditujukan untuk menyerang pihak lain (misalnya "anjir capek banget hari ini") bukan HS, walaupun bisa Abusive.

## 3. Label Abusive (Bahasa Kasar / Obscene)

### 3.1 Definisi Abusive
Abusive = 1 bila teks memuat kata kasar, makian, kata cabul, atau kata tidak pantas, baik ditujukan ke orang lain maupun tidak. Contoh kata: anjing, anjir, bangsat, bajingan, babi, goblok, goblog, bego, tolol, idiot, kampret, tai, setan, keparat, sialan, brengsek, monyet, geblek, edan, sinting, cabul, bencong, asu, jancuk, kontol, memek, ngentot, pantek, bacot.

### 3.2 Abusive tanpa HS
Makian sebagai ekspresi kaget atau candaan antar teman tanpa merendahkan pihak tertentu: "anjir seru bgt filmnya", "bangsat gue telat lagi". Label: Abusive = 1, HS = 0.

### 3.3 HS tanpa Abusive
Ujaran kebencian yang disampaikan dengan kata sopan: "kaum itu memang pantas diusir dari negeri ini". Label: HS = 1, HS_Group = 1, Abusive = 0.

## 4. Label HS_Individual (Target Individu)

### 4.1 Definisi HS_Individual
HS_Individual = 1 bila ujaran kebencian ditujukan kepada satu orang tertentu: tokoh publik yang disebut nama atau jabatannya (presiden, gubernur, menteri, ustadz tertentu), akun yang di-mention (USER), atau lawan bicara ("lu", "elo", "kamu", "dia"). Contoh: "USER otak lu dimana sih dasar tolol", "gubernur pembohong tukang ingkar janji".

### 4.2 Petunjuk target individu
Kata ganti orang kedua atau ketiga tunggal, nama orang, jabatan tunggal, atau mention USER yang diserang menandakan target individu.

## 5. Label HS_Group (Target Kelompok)

### 5.1 Definisi HS_Group
HS_Group = 1 bila ujaran kebencian ditujukan kepada kelompok: agama, ras, etnis, golongan politik, pendukung partai, gender, orientasi seksual, kelompok fisik, atau komunitas. Contoh: "cebong semua otaknya di dengkul", "kaum kafir harus dibasmi", "aseng antek komunis", "kadrun pada ngamuk".

### 5.2 Petunjuk target kelompok
Kata jamak atau penanda kelompok seperti kaum, umat, para, mereka, pendukung, partai, etnis, cina/aseng, kafir, cebong, kampret, kadrun, onta, banci menandakan target kelompok.

## 6. Teks Netral (Semua Label 0)

### 6.1 Definisi netral
Teks informatif, sapaan, ucapan terima kasih, doa, pertanyaan biasa, promosi, berita, atau opini tanpa hinaan dan tanpa kata kasar. Contoh: "semoga lancar acaranya kak", "harga cabai naik lagi minggu ini", "info lowongan kerja URL". Semua label = 0.

### 6.2 Kritik bukan hinaan
"Kebijakan ini kurang tepat karena merugikan petani" adalah kritik, bukan HS, selama tidak ada hinaan atau julukan merendahkan.

## 7. Kasus Ambigu dan Multi-label

### 7.1 Sarkasme
Sarkasme yang merendahkan pihak tertentu tetap HS ("pinter banget ya presiden kita sampe utang numpuk" bernada merendahkan = HS_Individual). Jika nada tidak jelas, utamakan bukti eksplisit pada teks.

### 7.2 Kutipan dan pelaporan
Teks yang mengutip ujaran kebencian untuk mengecam atau melaporkannya tidak otomatis HS. Teks yang mengutip lalu menyetujui tetap HS.

### 7.3 Kombinasi label
Contoh multi-label: "USER dasar goblok lu antek aseng" -> HS = 1, Abusive = 1, HS_Individual = 1, HS_Group = 0. "cebong anjing semua" -> HS = 1, Abusive = 1, HS_Individual = 0, HS_Group = 1.

## 8. Kebijakan Abstain

### 8.1 Kapan abstain
Sistem wajib abstain (abstain = true) bila teks di luar cakupan (Bagian 1.2) atau bila bukti tidak cukup dan keyakinan rendah. Abstain bukan berarti netral; abstain berarti sistem menolak memberi label otomatis dan teks diteruskan ke anotator manusia.

### 8.2 Alasan abstain
Alasan abstain dicatat: out_of_scope, insufficient_evidence, atau conflicting_evidence.

## 9. Aturan Bukti (Evidence Span)

### 9.1 Span wajib dari teks
Setiap label positif harus didukung span, yaitu potongan kata yang muncul persis di teks masukan. Span tidak boleh dikarang. Label positif tanpa span yang valid dianggap klaim tidak didukung (unsupported claim).

### 9.2 Contoh span
Teks "USER dasar goblok lu" -> span Abusive: "goblok"; span HS_Individual: "dasar goblok lu".
