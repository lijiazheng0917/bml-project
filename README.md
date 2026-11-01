# Predicting ICLR 2025 peer-review decisions with Bayesian models

Course project for CSE 5825 Bayesian Machine Learning (UConn), Jiazheng Li.

The project asks whether the information in peer reviews predicts the decision on an ICLR 2025
submission (reject, poster, spotlight, oral), and in particular whether reviews carry information
beyond the mean rating. It uses public OpenReview-derived data for 11,672 submissions.

| Folder | Content |
|---|---|
| `proj0/` | Project 0: proposal (PDF) |
| `proj1/` | Project 1: report (`proj1_report.pdf`) and code |

## Data

Three public releases (CC BY 4.0), joined by OpenReview paper ID:

- [ICLR Insights](https://github.com/papercopilot/iclr-insights) (Kargaran et al., 2025): each
  reviewer's initial and post-rebuttal rating, confidence, soundness, presentation, contribution.
- [ReviewerToo ICLR 2025](https://huggingface.co/datasets/demfier/reviewertoo-iclr2025): decision,
  primary area, abstract, review text, meta-review.
- [Paper Copilot](https://github.com/papercopilot/paperlists): submission status, used to identify
  withdrawn papers.

Raw and processed data files are not tracked. Small summaries used in the report are in
`proj1/data/processed/*.json`.

## Reproducing Project 1

The report is `proj1/proj1_report.pdf`. The scripts below rebuild the data tables and the figures
(written to `proj1/figures/`, which is not tracked).

```bash
# Python 3.10+ with these packages, e.g.
pip install pandas pyarrow matplotlib numpy scipy

cd proj1
bash code/00_download.sh    # download the raw data into data/raw (about 130 MB)
bash code/run_all.sh        # build the tables and figures
```

| Script | What it does |
|---|---|
| `code/00_download.sh` | Downloads the three raw files |
| `code/01_inspect.py` | Prints fields, label values and ID overlap of the raw files |
| `code/02_prepare.py` | Builds `reviews.parquet` and `papers.parquet`; fixes decision labels; 80/20 train/test split |
| `code/03_explore.py` | Figures 1-3 and summary numbers (training split only) |

The 20% test split is held out for the hypothesis tests in later project stages.
