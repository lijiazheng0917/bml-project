#!/usr/bin/env bash
# Download the raw data for Project 2 into data/raw (run from proj2/).
# Sources (all public): Paper Copilot paper lists, ICLR Insights, PRISM (review text).
set -euo pipefail
mkdir -p data/raw
cd data/raw
get() { [ -s "$2" ] && { echo "have $2"; return; }; echo "get  $2"; curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"; }

# Some years are stored with Git LFS (media URL), others as plain files (raw URL); try both.
for y in 2024 2025 2026; do
  f="papercopilot_iclr$y.json"
  get "https://media.githubusercontent.com/media/papercopilot/paperlists/main/iclr/iclr$y.json" "$f" 2>/dev/null \
    || get "https://raw.githubusercontent.com/papercopilot/paperlists/main/iclr/iclr$y.json" "$f"
done

INS=https://raw.githubusercontent.com/papercopilot/iclr-insights/main/dataset
get "$INS/iclr2024_threshold6_7407_reviewers.json" iclr2024_insights.json
get "$INS/iclr2025_threshold6_11677_reviewers.json" iclr2025_insights.json

# PRISM (thanhkt/prism-data on Hugging Face): review text and full paper text, ICLR 2024-2026
get https://huggingface.co/datasets/thanhkt/prism-data/resolve/main/papers.parquet prism_papers.parquet

date -u +"%Y-%m-%dT%H:%M:%SZ" > DOWNLOADED_AT.txt
ls -la
