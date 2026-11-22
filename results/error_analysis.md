# Analisis Penyebab Error (locked test, item in-scope dengan vektor label salah)

Taksonomi deterministik, diperiksa berurutan: **format/prompt** = output bukan JSON valid · **abstain** = in-scope di-abstain · **source** = dugaan noise label sumber (gold Abusive=0 padahal tweet memuat kata dari `abusive.csv` milik penyedia dataset, dan model memprediksi Abusive=1) · **tool** = (B6) tak satu pun tool call valid · **retrieval** = mayoritas label contoh ter-retrieve ≠ gold · **reasoning/model** = bukti retrieval mayoritas benar tetapi LLM salah · **no-retrieval** = konfigurasi tanpa retrieval (B0/B1). **adaptation** (efek LoRA) dibaca dari perubahan kolom reasoning B4 → B5.

| Config | #salah | format/prompt | abstain | source | tool | retrieval | reasoning/model | no-retrieval |
|---|---|---|---|---|---|---|---|---|
| B0 | 131 | 0 | 1 | 10 | 0 | 0 | 0 | 120 |
| B1 | 120 | 8 | 0 | 2 | 0 | 0 | 0 | 110 |
| B2 | 111 | 10 | 1 | 6 | 0 | 56 | 38 | 0 |
| B3 | 111 | 4 | 0 | 9 | 0 | 57 | 41 | 0 |
| B4 | 94 | 5 | 0 | 11 | 0 | 47 | 31 | 0 |
| B5 | 60 | 0 | 1 | 6 | 0 | 43 | 10 | 0 |
| B6 | 69 | 2 | 1 | 6 | 1 | 44 | 15 | 0 |

## Contoh kualitatif

### B6 benar, B5 salah (5 item) — 2 contoh

- **T06905** `USER Presiden Threshold adalah upaya rezim untuk mempertahankan kekuasaannya yg cenderung otoriter`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 1, 'Abusive': 0, 'HS_Individual': 1, 'HS_Group': 0} · B5={'HS': 1, 'Abusive': 0, 'HS_Individual': 1, 'HS_Group': 0} · B6={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · span B6=[]  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

- **T10209** `cina g punya perikemanusiaan makanya aq g suka cina di indonesia`  
  gold={'HS': 1, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 1}  
  B4={'HS': 1, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 1} · B5={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · B6={'HS': 1, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 1} · span B6=['cina']  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

### B5 benar, B4 salah (48 item) — 2 contoh

- **T10067** `USER mirip seperti hoax pokemon go = aku yahudi :')"`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 1, 'Abusive': 1, 'HS_Individual': 1, 'HS_Group': 0} · B5={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · B6={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · span B6=[]  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

- **T03007** `USER Ngobrol ayok ama aku yang cantiq dri cina ini wkwk`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 0, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 0} · B5={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · B6={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · span B6=[]  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

### Semua salah (43 item) — 2 contoh

- **T06535** `USER kalo USER gak laporin USER sekalian brarti fix klo satu gerbong... ð???ð???`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 1, 'Abusive': 1, 'HS_Individual': 1, 'HS_Group': 0} · B5={'HS': 1, 'Abusive': 1, 'HS_Individual': 1, 'HS_Group': 0} · B6={'HS': 1, 'Abusive': 1, 'HS_Individual': 1, 'HS_Group': 0} · span B6=['gerbong']  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

- **T02633** `USER Bravo pk jokowi tekan freeport utk kembalikan hak rakyat kelihatan sangat pro rakyat. Yang antek asing oknum penguasa, partai masa lalu yg jual saham freep`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 1, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 1} · B5={'HS': 1, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 1} · B6={'HS': 1, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 1} · span B6=['jokowi', 'rakyat']  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

