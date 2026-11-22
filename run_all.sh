#!/usr/bin/env bash
# Menjalankan seluruh pipeline eksperimen secara berurutan (reproducible).
# Windows (Git Bash) / Linux / Kaggle / Colab: bash run_all.sh
set -e
cd "$(dirname "$0")"
# Python: venv Windows (laptop) bila ada, selain itu python sistem (Kaggle/Colab/Linux)
if [ -z "$PY" ]; then
  if [ -x ./.venv/Scripts/python ]; then PY=./.venv/Scripts/python; elif [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python; else PY=python; fi
fi
echo "Python: $PY"
export PYTHONIOENCODING=utf-8
mkdir -p logs
[ -f data/processed/test.jsonl ] || $PY -m tbp.data_prep              > logs/data_prep.log 2>&1
[ -f adapters/lora_b5/adapter_config.json ] || $PY -m tbp.train_lora  > logs/train_lora.log 2>&1
for st in retrieval dev test ablations repeat ood; do
  if ! grep -q "\"$st\"\|stage $st selesai" logs/stage_$st.log 2>/dev/null; then
    $PY -m tbp.run_experiments $st > logs/stage_$st.log 2>&1
  fi
  echo "stage $st OK"
done
$PY -m tbp.redteam     > logs/redteam.log 2>&1 && echo "redteam OK"
$PY -m tbp.make_report > logs/make_report.log 2>&1 && echo "report OK"
echo ALL_DONE
