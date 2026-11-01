#!/bin/bash
# Rebuild everything for Project 1 on a compute node:
#   srun -p general -A jhf24001 -c2 --mem=8G -t 00:30:00 bash code/run_all.sh
# (Run 00_download.sh first if data/raw is empty.)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=~/miniforge3/envs/bml/bin/python
$PY code/02_prepare.py > data/processed/prepare_log.txt
$PY code/03_explore.py > data/processed/explore_log.txt
cd report && ~/miniforge3/envs/tex/bin/tectonic -X compile proj1_report.tex > latex_log.txt 2>&1
echo "done: report/proj1_report.pdf"
