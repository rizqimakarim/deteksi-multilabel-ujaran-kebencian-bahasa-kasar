"""Membangun paket Kaggle:
  dist/kaggle_dataset/tbp_project.zip + dataset-metadata.json      (Dataset PRIVATE berisi kode + data mentah)
  dist/kaggle_run/   (notebook 1: jalankan seluruh eksperimen di GPU T4)
  dist/kaggle_demo/  (notebook 2: demo Streamlit + API + link publik cloudflared)
Jalankan dari root proyek:  python tools/make_kaggle_package.py <kaggle_username>
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
USER = sys.argv[1] if len(sys.argv) > 1 else "YOUR_KAGGLE_USERNAME"
DATASET_SLUG = "case-b-multi-label-hate-speech-indonesian-twitter"
RUN_SLUG = "case-b-multilabel-hate-speech-indonesian-twitter"
DEMO_SLUG = "streamlit-case-b-hate-speech-indonesian-twitter"

INCLUDE = ["tbp", "tools", "app_streamlit.py", "run_all.sh", "start_demo.sh", "requirements.txt", ".streamlit",
           "data/raw", "data/guidelines", "data/ood", "human_review", "docs/ADR.md", "docs/screenshots",
           "docs/references/Ibrohim_Budi_2019_W19-3506.pdf", "README.md", "LAPORAN.md", "RINGKASAN_PROYEK.md",
           "NLP_IR.pdf", "archive", "KAGGLE_PANDUAN.md", "ALUR_END_TO_END.md"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", "lora_b5_adapter"}   # adapter laptop lama tidak diunggah


def build_zip(dst):
    n = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            if not p.exists():
                print("  (lewati, tidak ada)", item)
                continue
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                if SKIP_PARTS & set(f.parts):
                    continue
                z.write(f, Path("tbp_project") / f.relative_to(ROOT))
                n += 1
    return n


# ----------------------------------------------------------------------------- notebook helpers
def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def notebook(cells):
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:03d}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


FIND_PROJECT = r'''import glob, os, shutil, subprocess, zipfile, sys, time, json
WORK = "/kaggle/working/tbp_project"

def find_project():
    """Cari folder proyek di /kaggle/input (zip otomatis diekstrak Kaggle; bila belum, ekstrak sendiri)."""
    hits = glob.glob("/kaggle/input/**/tbp/config.py", recursive=True)
    hits.sort(key=lambda h: "/datasets/" not in h)      # utamakan dataset (kode & laporan terbaru), bukan output notebook
    if hits:
        return os.path.dirname(os.path.dirname(hits[0]))
    zips = glob.glob("/kaggle/input/**/tbp_project.zip", recursive=True)
    if zips:
        zipfile.ZipFile(zips[0]).extractall("/kaggle/tmp_extract")
        return "/kaggle/tmp_extract/tbp_project"
    raise FileNotFoundError("Dataset proyek tidak ditemukan. Tambahkan dataset 'Case B: Multi-label Hate Speech Indonesian Twitter' lewat Add Input.")

src = find_project()
print("Sumber proyek:", src)
if not os.path.exists(WORK):
    shutil.copytree(src, WORK)
os.chdir(WORK)
print("Folder kerja:", os.getcwd())
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
'''

PIP = r'''!pip install -q transformers==5.23.0 peft==0.22.0 accelerate==1.16.0 sentence-transformers==6.3.0 rank-bm25==0.2.2 {extra}
import torch, transformers, peft
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("transformers", transformers.__version__, "| peft", peft.__version__)
'''

RUN = r'''# Menjalankan SELURUH pipeline: data_prep -> LoRA -> retrieval -> dev -> test (B0-B6) -> ablation -> repeat -> OOD -> red-team -> tabel
# Progres tiap tahap tampil di bawah; log detail ada di tbp_project/logs/
import subprocess, glob, os
for d in ["data/processed", "results", "adapters", "logs"]:      # pastikan run bersih
    shutil.rmtree(d, ignore_errors=True)
t0 = time.time()
p = subprocess.Popen(["bash", "run_all.sh"], env={**os.environ, "PY": sys.executable, "PYTHONIOENCODING": "utf-8"},
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(f"[{(time.time()-t0)/60:5.1f} menit] {line}", end="", flush=True)
p.wait()
if p.returncode != 0:
    logs = sorted(glob.glob("logs/*.log"), key=os.path.getmtime)
    print("\n=== GAGAL. 60 baris terakhir dari", logs[-1], "===")
    print("".join(open(logs[-1], encoding="utf-8", errors="replace").readlines()[-60:]))
    raise SystemExit(p.returncode)
print(f"\nSELESAI dalam {(time.time()-t0)/60:.1f} menit")
'''

SUMMARY = r'''R = json.load(open("results/results.json"))
print("Parent B6 (dipilih di dev):", R["dev_selection"])
print(f"\n{'Config':8s} {'Macro-F1':>9s} {'Evid':>6s} {'OOS-abst':>9s} {'p95 ms':>8s}")
for c, m in R["main_test"].items():
    print(f"{c:8s} {m['macro_f1']:9.3f} {m['evidence_support'] or 0:6.2f} {m['oos_abstain_recall'] or 0:9.2f} {m['cost']['latency_p95_ms']:8.0f}")
print("\nDelta bootstrap:", json.dumps(R["deltas_bootstrap"], indent=0)[:1500])
print("\nOOD:", R.get("ood"))
print("\nRed-team:", json.load(open("results/redteam.json", encoding="utf-8"))["summary"])
'''

PACK = r'''# Satu file zip berisi semua hasil (untuk diunduh / dipakai notebook demo)
out = "/kaggle/working/tbp_output.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ["results", "adapters", "data/processed", "logs", "human_review"]:
        for f in glob.glob(f"{d}/**/*", recursive=True):
            if os.path.isfile(f):
                z.write(f, f)
env = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
json.dump(env, open("results/environment_kaggle.json", "w"), indent=1)
print("Output:", out, round(os.path.getsize(out)/1e6, 1), "MB")
print(env)
'''

DEMO_RESTORE = r'''# Ambil hasil run (notebook 1) dari Input: tbp_output.zip atau folder hasil
cands = glob.glob("/kaggle/input/**/tbp_output.zip", recursive=True)
if cands:
    zipfile.ZipFile(cands[0]).extractall(WORK)
    print("Hasil dipulihkan dari", cands[0])
else:
    res = glob.glob("/kaggle/input/**/tbp_project/results/results.json", recursive=True)
    if not res:
        raise FileNotFoundError("Output notebook 'Case B: Multilabel Hate Speech Indonesian Twitter' belum ditambahkan lewat Add Input.")
    base = os.path.dirname(os.path.dirname(res[0]))
    for d in ["results", "adapters", "data/processed", "logs"]:
        if os.path.exists(f"{base}/{d}"):
            shutil.copytree(f"{base}/{d}", d, dirs_exist_ok=True)
    print("Hasil dipulihkan dari", base)
print("results.json ada:", os.path.exists("results/results.json"), "| adapter:", os.path.exists("adapters/lora_b5/adapter_config.json"))
'''

DEMO_START = r'''# Nyalakan API (FastAPI :8000) dan UI Streamlit (:8501) di background
os.makedirs("logs", exist_ok=True)
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
api = subprocess.Popen([sys.executable, "-m", "uvicorn", "tbp.serve:app", "--port", "8000"],
                       stdout=open("logs/serve_demo.log", "w"), stderr=subprocess.STDOUT, env=env)
import urllib.request
def up(url):
    try:
        urllib.request.urlopen(url, timeout=2); return True
    except Exception:
        return False
for _ in range(120):
    if up("http://localhost:8000/health"): break
    time.sleep(2)
print("API siap:", up("http://localhost:8000/health"))
ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app_streamlit.py", "--server.port", "8501",
                       "--server.headless", "true"], stdout=open("logs/streamlit.log", "w"), stderr=subprocess.STDOUT, env=env)
for _ in range(60):
    if up("http://localhost:8501/_stcore/health"): break
    time.sleep(2)
print("Streamlit siap:", up("http://localhost:8501/_stcore/health"))
'''

DEMO_TUNNEL = r'''# Link publik sementara via Cloudflare Quick Tunnel (tanpa akun). Link berubah setiap sesi.
import re, stat, urllib.request
cf = "/kaggle/working/cloudflared"
if not os.path.exists(cf):
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cf)
    os.chmod(cf, os.stat(cf).st_mode | stat.S_IEXEC)
tun = subprocess.Popen([cf, "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url = None
t0 = time.time()
while time.time() - t0 < 90 and not url:
    line = tun.stdout.readline()
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
print("=" * 70)
print("BUKA DEMO DI:", url or "(gagal membuat tunnel - cek Internet ON)")
print("Bagikan link ini ke anggota tim. Link aktif selama sesi notebook ini menyala.")
print("=" * 70)
'''

DEMO_KEEP = r'''# Mode "Interactive" (dibuka di browser): sel ini menjaga sesi tetap aktif selama demo (hentikan dengan tombol Stop).
# Klasifikasi pertama butuh ±30 detik (memuat model), berikutnya ±15–40 detik untuk B0–B6 sekaligus.
# Mode "Batch" (Save & Run All / API): lakukan uji otomatis lalu selesai, agar notebook tidak menyala 12 jam.
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Interactive") == "Batch":
    req = urllib.request.Request("http://localhost:8000/compare", data=json.dumps({"text": "USER dasar goblok lu, cebong semua sama aja"}).encode(),
                                 headers={"Content-Type": "application/json"})
    rows = json.loads(urllib.request.urlopen(req, timeout=900).read())
    for r in rows:
        print(r["config"], "ABSTAIN" if r["abstain"] else r["labels"], r["evidence_spans"], f"{r['latency_ms']:.0f} ms")
    try:
        page = urllib.request.urlopen(url, timeout=60).status if url else None
    except Exception as e:
        page = f"error: {e}"
    print("SELF-TEST:", "OK" if len(rows) == 7 and up("http://localhost:8501/_stcore/health") and page == 200 else "PERLU DICEK",
          "| tunnel HTTP", page)
    for p_ in (tun, ui, api):
        p_.terminate()
else:
    try:
        while True:
            time.sleep(60)
            print(time.strftime("%H:%M"), "| API", api.poll() is None, "| UI", ui.poll() is None, "| tunnel", tun.poll() is None, flush=True)
    except KeyboardInterrupt:
        print("Dihentikan.")
'''


def run_notebook():
    sys.path.insert(0, str(ROOT / 'tools'))
    import story_notebook
    return notebook(story_notebook.build(FIND_PROJECT, PIP.format(extra='matplotlib')))


def run_notebook_old():
    return notebook([
        md("# Case B — Notebook 1: Jalankan seluruh eksperimen (GPU T4)\n\n"
           "**Kelompok Rizqi — Case B: Multi-label Hate Speech & Abusive Language Detection (Twitter Indonesia)**\n\n"
           "Notebook ini menjalankan pipeline lengkap dari nol: persiapan data → LoRA → B0–B6 → ablation A1–A7 → "
           "repeatability → OOD → red-team → tabel laporan.\n\n"
           "**Pengaturan (panel kanan):** Accelerator = **GPU T4**, Internet = **On**, Input = dataset **Case B: Multi-label Hate Speech Indonesian Twitter**. "
           "Jalankan dengan **Save Version → Save & Run All** (±2–3 jam). Semua hasil tersimpan di Output (`tbp_project/` dan `tbp_output.zip`)."),
        code(FIND_PROJECT),
        code(PIP.format(extra="")),
        md("## Jalankan pipeline"),
        code(RUN),
        md("## Ringkasan hasil"),
        code(SUMMARY),
        md("## Paketkan output"),
        code(PACK),
    ])


def demo_notebook():
    return notebook([
        md("# Case B — Notebook 2: Demo Streamlit (B0–B6, Hasil Eksperimen, Laporan)\n\n"
           "**Cara pakai (sesi interaktif):**\n"
           "1. Klik **Edit** / buka notebook ini.\n"
           "2. Panel kanan: Accelerator = **GPU T4**, Internet = **On**.\n"
           "3. **Add Input**: dataset **Case B: Multi-label Hate Speech Indonesian Twitter** dan output notebook **Case B: Multilabel Hate Speech Indonesian Twitter** (notebook eksperimen).\n"
           "4. Klik **Run All**. Di bawah sel *tunnel* muncul link `https://....trycloudflare.com` — buka & bagikan ke tim.\n\n"
           "Link hanya aktif selama sesi ini menyala."),
        code(FIND_PROJECT),
        code(PIP.format(extra="streamlit==1.67.0 altair==6.3.0 fastapi uvicorn")),
        code(DEMO_RESTORE),
        code(DEMO_START),
        code(DEMO_TUNNEL),
        code(DEMO_KEEP),
    ])


def main():
    if DIST.exists():
        shutil.rmtree(DIST)
    ds = DIST / "kaggle_dataset"
    ds.mkdir(parents=True)
    n = build_zip(ds / "tbp_project.zip")
    json.dump({"title": "Case B: Multi-label Hate Speech Indonesian Twitter", "id": f"{USER}/{DATASET_SLUG}",
               "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, open(ds / "dataset-metadata.json", "w"), indent=2)
    for slug, title, nb, srcs in [(RUN_SLUG, "Case B: Multilabel Hate Speech Indonesian Twitter", run_notebook(), []),
                                  (DEMO_SLUG, "Streamlit - Case B: Hate Speech Indonesian Twitter", demo_notebook(), [f"{USER}/{RUN_SLUG}"])]:
        d = DIST / slug
        d.mkdir(parents=True)
        fn = f"{slug}.ipynb"
        json.dump(nb, open(d / fn, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        json.dump({"id": f"{USER}/{slug}", "title": title, "code_file": fn, "language": "python", "kernel_type": "notebook",
                   "is_private": "true", "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
                   "machine_shape": "", "dataset_sources": [f"{USER}/{DATASET_SLUG}"], "competition_sources": [],
                   "kernel_sources": srcs, "model_sources": []}, open(d / "kernel-metadata.json", "w"), indent=2)
        shutil.copy(d / fn, ROOT / "kaggle" / fn) if (ROOT / "kaggle").exists() else None
    print(f"Paket: {ds / 'tbp_project.zip'} ({n} file, {(ds / 'tbp_project.zip').stat().st_size / 1e6:.1f} MB)")
    print("Notebook:", [str(p.relative_to(ROOT)) for p in DIST.glob('*/*.ipynb')])


if __name__ == "__main__":
    main()
