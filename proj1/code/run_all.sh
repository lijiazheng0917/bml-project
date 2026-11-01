#!/bin/bash
# Rebuild the Project 1 tables and figures:
#   bash code/run_all.sh
# Run code/00_download.sh first if data/raw is empty.
# Uses the `python` of the active environment; set PYTHON=/path/to/python to use another one.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
mkdir -p data/processed
"$PY" code/02_prepare.py > data/processed/prepare_log.txt
"$PY" code/03_explore.py > data/processed/explore_log.txt

# The LaTeX source of the report is kept outside the repo; build it only if it is present.
# Set TECTONIC=/path/to/tectonic if tectonic is not on the PATH.
if [ -f report/proj1_report.tex ]; then
  (cd report && "${TECTONIC:-tectonic}" -X compile proj1_report.tex > latex_log.txt 2>&1)
  cp report/proj1_report.pdf proj1_report.pdf
  echo "done: tables, figures/ and proj1_report.pdf"
else
  echo "done: tables and figures/"
fi
