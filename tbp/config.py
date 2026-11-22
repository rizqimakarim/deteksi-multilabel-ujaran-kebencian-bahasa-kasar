"""Konfigurasi global proyek TBP (Use case B: Evidence-Aware Multi-label Hate Speech & Abusive Detection)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
GUIDE_DIR = DATA / "guidelines"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"
ADAPTERS = ROOT / "adapters"
LOGS = ROOT / "logs"
for p in (PROC, RESULTS, RUNS, ADAPTERS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

LABELS = ["HS", "Abusive", "HS_Individual", "HS_Group"]

# Base generator yang SAMA untuk B0-B6 (kewajiban spesifikasi)
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMB_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

SEED = 42
N_TEST_INSCOPE = 200
N_DEV_INSCOPE = 100
REPEAT_SUBSET = 40          # subset test untuk 3 run independen
REPEAT_SEEDS = [11, 22, 33]
MAX_NEW_TOKENS = 80

# Budget default (A7 membandingkan dengan budget besar)
BUDGET_SMALL = {"k_examples": 4, "k_guides": 1, "max_ctx_chars": 1400}
BUDGET_LARGE = {"k_examples": 10, "k_guides": 2, "max_ctx_chars": 4200}

# Tanggal acuan (as_of) untuk pemilihan versi guideline
DEFAULT_AS_OF = "2026-11-20"
