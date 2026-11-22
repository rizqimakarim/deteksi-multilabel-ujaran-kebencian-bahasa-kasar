# Analisis Penyebab Error (locked test, item in-scope dengan vektor label salah)

Taksonomi deterministik, diperiksa berurutan: **format/prompt** = output bukan JSON valid · **abstain** = in-scope di-abstain · **source** = dugaan noise label sumber (gold Abusive=0 padahal tweet memuat kata dari `abusive.csv` milik penyedia dataset, dan model memprediksi Abusive=1) · **tool** = (B6) tak satu pun tool call valid · **retrieval** = mayoritas label contoh ter-retrieve ≠ gold · **reasoning/model** = bukti retrieval mayoritas benar tetapi LLM salah · **no-retrieval** = konfigurasi tanpa retrieval (B0/B1). **adaptation** (efek LoRA) dibaca dari perubahan kolom reasoning B4 → B5.

| Config | #salah | format/prompt | abstain | source | tool | retrieval | reasoning/model | no-retrieval |
|---|---|---|---|---|---|---|---|---|
| B0 | 131 | 0 | 1 | 10 | 0 | 0 | 0 | 120 |
| B1 | 119 | 11 | 0 | 3 | 0 | 0 | 0 | 105 |
| B2 | 109 | 8 | 1 | 7 | 0 | 54 | 39 | 0 |
| B3 | 114 | 5 | 0 | 9 | 0 | 56 | 44 | 0 |
| B4 | 97 | 4 | 0 | 11 | 0 | 48 | 34 | 0 |
| B5 | 67 | 0 | 1 | 7 | 0 | 47 | 12 | 0 |
| B6 | 65 | 0 | 1 | 6 | 0 | 41 | 17 | 0 |

## Contoh kualitatif

### B6 benar, B5 salah (14 item) — 2 contoh

- **T06535** `USER kalo USER gak laporin USER sekalian brarti fix klo satu gerbong... ð???ð???`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 1, 'Abusive': 1, 'HS_Individual': 1, 'HS_Group': 0} · B5={'HS': 1, 'Abusive': 1, 'HS_Individual': 1, 'HS_Group': 0} · B6={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · span B6=[]  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

- **T00732** `USER 6 jam perjalanan bang, tapi di sini adem Wkwkwk ((( naik onta )))`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 0, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 0} · B5={'HS': 0, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 0} · B6={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · span B6=[]  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

### B5 benar, B4 salah (46 item) — 2 contoh

- **T10067** `USER mirip seperti hoax pokemon go = aku yahudi :')"`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 1, 'Abusive': 1, 'HS_Individual': 1, 'HS_Group': 0} · B5={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · B6={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · span B6=[]  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

- **T03007** `USER Ngobrol ayok ama aku yang cantiq dri cina ini wkwk`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 0, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 0} · B5={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · B6={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0} · span B6=[]  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'ok'), ('lexicon_lookup', 'budget_overrun')]

### Semua salah (40 item) — 2 contoh

- **T02633** `USER Bravo pk jokowi tekan freeport utk kembalikan hak rakyat kelihatan sangat pro rakyat. Yang antek asing oknum penguasa, partai masa lalu yg jual saham freep`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 1, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 1} · B5={'HS': 1, 'Abusive': 1, 'HS_Individual': 1, 'HS_Group': 0} · B6={'HS': 1, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 1} · span B6=['asing', 'jokowi']  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'invalid_args'), ('get_policy', 'budget_overrun')]

- **T08363** `USER Berasa punya monyet :v .g`  
  gold={'HS': 0, 'Abusive': 0, 'HS_Individual': 0, 'HS_Group': 0}  
  B4={'HS': 0, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 0} · B5={'HS': 0, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 0} · B6={'HS': 0, 'Abusive': 1, 'HS_Individual': 0, 'HS_Group': 0} · span B6=['monyet']  
  tool B6: [('search_examples', 'ok'), ('get_policy', 'invalid_args'), ('get_policy', 'budget_overrun')]

