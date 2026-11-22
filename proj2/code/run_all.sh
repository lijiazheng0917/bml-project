#!/usr/bin/env bash
# Rebuild all Project 2 results. Run from proj2/ after Project 1's data step
# (proj1/data/processed/papers.parquet holds the fixed 2025 train/test split).
#
#   PYTHON=python bash code/run_all.sh          # everything except the paid LLM calls
#   OPENROUTER_API_KEY=... LLM=1 bash code/run_all.sh   # also (re)code the reviews with the LLM
#
# The LLM step costs about US$21 (2024+2025) and US$16 (2026) with DeepSeek V4.1 Flash;
# without LLM=1 the script expects data/processed/llm_codes*.parquet to exist.
set -euo pipefail
PY=${PYTHON:-python}
D=data/processed/decision
mkdir -p "$D" figures

bash code/00_download.sh
$PY code/01_inspect.py
$PY code/02_prepare.py
$PY code/02b_prepare_2026.py

if [ "${LLM:-0}" = "1" ]; then
  $PY code/10_llm_pilot.py --models "deepseek/deepseek-v4.1-flash-1017,z-ai/glm-5.3-flash" --n 200 --repeat 50
  $PY code/11_llm_extract.py --model deepseek/deepseek-v4.1-flash-1017 --workers 64
  $PY code/11_llm_extract.py --model deepseek/deepseek-v4.1-flash-1017 --workers 64 --suffix _2026
fi
$PY code/07_llm_check.py
$PY code/12_llm_style.py

# Decision models, ICLR 2024 + 2025 (NUTS, 4 chains run one after another)
fit() { $PY -W ignore code/03_decision.py "$@" --warmup 1000 --samples 1000; }
for m in mean votes free rebut conf cons; do fit --model $m > "$D/log_$m.txt" 2>&1; done
fit --model text --text-col text_z > "$D/log_text.txt" 2>&1
for y in 2024 2025; do
  for m in mean free conf cons; do fit --model $m --train $y --test 2024,2025 > "$D/log_${m}_train$y.txt" 2>&1; done
  fit --model text --text-col text_z --train $y --test 2024,2025 > "$D/log_text_train$y.txt" 2>&1
done

# Decision models, ICLR 2026 (separate rating scale; final report)
for m in mean free conf; do BML_DATA=_2026 fit --model $m --train 2026 --test 2026 > "$D/log_${m}_2026.txt" 2>&1; done
BML_DATA=_2026 fit --model text --text-col text_z --train 2026 --test 2026 > "$D/log_text_2026.txt" 2>&1

$PY code/05_simulate.py
$PY code/08_baselines.py
$PY -W ignore code/04_summarize.py > "$D/summary.txt"
BML_DATA=_2026 $PY -W ignore code/04_summarize.py > "$D/summary_2026.txt"
$PY code/09_importance.py
$PY code/06_figures.py
echo "done: tables in $D, figures in figures/"
