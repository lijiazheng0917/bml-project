#!/bin/bash
# Download the raw data for Project 1 into data/raw (about 130 MB):
#   bash code/00_download.sh
set -euo pipefail
cd "$(dirname "$0")/../data/raw"

# Reviewer-level initial;final scores (Kargaran et al., ICLR Insights)
curl -fsSL -o iclr2025_insights.json \
  https://raw.githubusercontent.com/papercopilot/iclr-insights/main/dataset/iclr2025_threshold6_11677_reviewers.json

# Paper-level decisions, areas, review text, meta-reviews (ReviewerToo full release)
curl -fsSL -o reviewertoo_iclr2025_full.parquet \
  https://huggingface.co/datasets/demfier/reviewertoo-iclr2025/resolve/main/iclr2025_full.parquet

# Paper Copilot ICLR 2025 list (submission status, incl. withdrawals)
curl -fsSL -o papercopilot_iclr2025.json \
  https://media.githubusercontent.com/media/papercopilot/paperlists/main/iclr/iclr2025.json

date -u +"downloaded %Y-%m-%dT%H:%MZ" > DOWNLOADED_AT.txt
ls -la
