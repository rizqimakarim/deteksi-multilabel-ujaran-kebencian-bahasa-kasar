"""Serving: FastAPI (API + UI web sederhana), observability (log JSONL + trace), fallback.
Jalankan:  .venv\\Scripts\\python -m uvicorn tbp.serve:app --port 8000   lalu buka http://127.0.0.1:8000
"""
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import DEFAULT_AS_OF, LABELS, LOGS
from .pipelines import Pipeline, lexicon_hits, neighbor_vote
from .retrieval import get_index
from .serving_guard import redact_pii, sanitize_input

app = FastAPI(title="TBP Evidence-Aware Multi-label Hate Speech API", version="1.0")
LOG = LOGS / "serving.jsonl"
_PIPES = {}


def pipe(cfg):
    if cfg not in _PIPES:
        if cfg == "B6":
            from .agent import AgentPipeline
            from .run_experiments import parent
            _PIPES[cfg] = AgentPipeline(parent(), use_tools=True)
        else:
            _PIPES[cfg] = Pipeline(cfg)
    return _PIPES[cfg]


class Req(BaseModel):
    text: str
    config: str = "B6"
    as_of: str = DEFAULT_AS_OF
    force_fallback: bool = False   # untuk demo/uji jalur fallback


def fallback(text):
    """Fallback tanpa LLM: kNN vote dari contoh train (hybrid retrieval) + span lexicon."""
    ex, _ = get_index("examples").search(text, k=7, mode="hybrid", flt={"doc_type": "example"})
    v = neighbor_vote(ex)
    pred = {l: int(v[l] >= 0.5) for l in LABELS}
    return {"pred": pred, "scores": {l: round(v[l], 3) for l in LABELS}, "evidence": lexicon_hits(text)[:3],
            "abstain": False, "abstain_reason": None, "retrieved_ids": [d["doc_id"] for d in ex]}


@app.post("/classify")
def classify(req: Req):
    trace_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    text, note = sanitize_input(req.text)
    text = redact_pii(text)                         # PII tidak pernah masuk model/log
    cfg = req.config if req.config in ("B0", "B1", "B2", "B3", "B4", "B5", "B6") else "B6"
    used_fallback, err = False, None
    try:
        if req.force_fallback:
            raise RuntimeError("force_fallback=true (simulasi LLM tidak tersedia)")
        p = pipe(cfg)
        p.as_of = req.as_of
        r = p.run([{"id": trace_id, "text": text}])[0]
        if not r.get("parse_ok", True):
            raise ValueError("output LLM tidak valid terhadap kontrak JSON")
    except Exception as e:  # fallback
        used_fallback, err = True, str(e)[:200]
        r = fallback(text)
    resp = {"trace_id": trace_id, "config": cfg, "labels": r["pred"], "scores": r.get("scores"),
            "evidence_spans": r.get("evidence", []), "abstain": r["abstain"], "abstain_reason": r.get("abstain_reason"),
            "guideline_version": (r.get("guideline_versions") or ["1.0"]), "as_of": req.as_of,
            "retrieved_ids": r.get("retrieved_ids", [])[:8], "tool_log": r.get("tool_log"),
            "fallback_used": used_fallback, "input_note": note, "redacted_input": text,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **resp, "error": err,
                            "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens")},
                           ensure_ascii=False, default=str) + "\n")
    return resp


class CompareReq(BaseModel):
    text: str
    as_of: str = DEFAULT_AS_OF


@app.post("/compare")
def compare(req: CompareReq):
    """Jalankan B0-B6 berurutan untuk teks yang sama (untuk demo perbandingan).
    Urutan B0-B4 (model dasar) lalu B5-B6 (model + LoRA) agar model hanya berganti sekali."""
    out = []
    for cfg in ("B0", "B1", "B2", "B3", "B4", "B5", "B6"):
        out.append(classify(Req(text=req.text, config=cfg, as_of=req.as_of)))
    return out


@app.get("/health")
def health():
    return {"status": "ok", "labels": LABELS}


@app.get("/logs")
def logs(n: int = 20):
    if not LOG.exists():
        return []
    return [json.loads(l) for l in open(LOG, encoding="utf-8").read().splitlines()[-n:]]


UI = """<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deteksi Ujaran Kebencian</title><style>
body{font-family:system-ui,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}
textarea{width:100%;height:90px;font-size:15px;padding:8px;box-sizing:border-box}
button{padding:8px 18px;font-size:15px;margin-top:8px;cursor:pointer}
.lab{display:inline-block;padding:4px 10px;margin:4px;border-radius:12px;background:#e5e7eb}
.on{background:#dc2626;color:#fff}.box{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin-top:14px}
pre{white-space:pre-wrap;font-size:12px;background:#f3f4f6;padding:8px;border-radius:6px}
</style></head><body>
<h2>Deteksi Multi-label Ujaran Kebencian &amp; Bahasa Kasar (Bahasa Indonesia)</h2>
<textarea id="t" placeholder="Tulis tweet...">USER dasar goblok lu, cebong semua sama aja</textarea>
<div><button onclick="cmp()">Klasifikasi</button>
<span style="color:#666;font-size:13px">&nbsp;menampilkan hasil B0&ndash;B6 sekaligus (~15&ndash;40 detik)</span></div>
<div id="o"></div>
<script>
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function go(){
 document.getElementById('o').innerHTML='<div class=box>memproses...</div>';
 const r=await fetch('/classify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('t').value,config:document.getElementById('c').value})});
 const j=await r.json();
 let h='<div class=box>';
 for(const [k,v] of Object.entries(j.labels)) h+=`<span class="lab ${v?'on':''}">${k}: ${v}${j.scores?' ('+j.scores[k]+')':''}</span>`;
 h+=`<p><b>Abstain:</b> ${j.abstain} ${j.abstain_reason||''} &nbsp; <b>Fallback:</b> ${j.fallback_used} &nbsp; <b>Latensi:</b> ${j.latency_ms} ms</p>`;
 h+=`<p><b>Bukti (span teks):</b> ${(j.evidence_spans||[]).map(s=>'<mark>'+esc(s)+'</mark>').join(', ')||'-'}</p>`;
 h+=`<p><b>Pedoman versi:</b> ${j.guideline_version} (as_of ${j.as_of}) &nbsp; <b>trace_id:</b> ${j.trace_id}</p>`;
 h+='<details><summary>Trace lengkap</summary><pre>'+esc(JSON.stringify(j,null,1))+'</pre></details></div>';
 document.getElementById('o').innerHTML=h;}
const DESC={B0:'AI polos',B1:'+ definisi & contoh',B2:'+ cari tweet mirip (RAG)',B3:'+ pencarian hybrid',B4:'+ cek ulang & abstain',B5:'+ LoRA (dilatih)',B6:'+ agent & tool'};
async function cmp(){
 const o=document.getElementById('o');
 o.innerHTML='<div class=box>menjalankan B0&ndash;B6 berurutan... (~15&ndash;40 detik)</div>';
 const t0=Date.now();
 const r=await fetch('/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('t').value})});
 const rows=await r.json();
 const L=['HS','Abusive','HS_Individual','HS_Group'];
 let h='<div class=box><table style="width:100%;border-collapse:collapse;font-size:14px"><tr style="text-align:left;border-bottom:2px solid #ddd"><th>Versi</th><th>Keterangan</th>'+L.map(l=>'<th>'+l+'</th>').join('')+'<th>Bukti</th><th>ms</th></tr>';
 for(const j of rows){
  h+=`<tr style="border-bottom:1px solid #eee"><td><b>${j.config}</b></td><td style="color:#666">${DESC[j.config]}</td>`;
  if(j.abstain) h+=`<td colspan="4"><span class="lab">ABSTAIN: ${esc(j.abstain_reason||'')}</span></td>`;
  else h+=L.map(l=>`<td><span class="lab ${j.labels[l]?'on':''}">${j.labels[l]}</span></td>`).join('');
  h+=`<td>${(j.evidence_spans||[]).map(s=>'<mark>'+esc(s)+'</mark>').join(', ')||'-'}</td><td>${Math.round(j.latency_ms)}</td></tr>`;
 }
 h+=`</table><p style="color:#666;font-size:13px">Total ${((Date.now()-t0)/1000).toFixed(1)} detik. Merah = 1 (label aktif). Nilai macro-F1 di ujian: B0 0,16 &middot; B1 0,10 &middot; B2 0,43 &middot; B3 0,47 &middot; B4 0,55 &middot; B5 0,64 &middot; B6 0,67</p></div>`;
 o.innerHTML=h;}
const qp=new URLSearchParams(location.search);
if(qp.get('q')){document.getElementById('t').value=qp.get('q');cmp();}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return UI
