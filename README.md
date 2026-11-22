# How ICLR area chairs turn peer reviews into decisions: Bayesian models

Course project for CSE 5825 Bayesian Machine Learning (UConn), Jiazheng Li.

The project asks how the information in peer reviews determines the decision on an ICLR
submission (reject, poster, spotlight, oral), and in particular whether reviews carry information
beyond the mean rating. Projects 0 and 1 use ICLR 2025; Project 2 adds ICLR 2024 and the review
text, coded by a large language model.

| Folder | Content |
|---|---|
| `proj0/` | Project 0: proposal (PDF) |
| `proj1/` | Project 1: report (`proj1_report.pdf`) and code |
| `proj2/` | Project 2 (milestone): report (`proj2_report.pdf`), code, and small result summaries |

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

## Reproducing Project 2

The report is `proj2/proj2_report.pdf`. Project 2 reuses the 2025 train/test split of Project 1,
so run the Project 1 data step first (`proj1/code/02_prepare.py`).

```bash
pip install pandas pyarrow numpy scipy matplotlib "jax[cpu]" numpyro arviz scikit-learn

cd proj2
PYTHON=python bash code/run_all.sh                          # data, models, tables, figures
OPENROUTER_API_KEY=... LLM=1 PYTHON=python bash code/run_all.sh   # also redo the LLM coding
```

Additional data: [Paper Copilot](https://github.com/papercopilot/paperlists) lists for ICLR
2024 and 2026, ICLR Insights for 2024, and the review text from
[PRISM](https://huggingface.co/datasets/thanhkt/prism-data). The LLM step calls DeepSeek V4.1
Flash through OpenRouter and costs about US$21 for 2024 and 2025; without `LLM=1` the script
expects the coded reviews (`data/processed/llm_codes*.parquet`) to exist.

| Script | What it does |
|---|---|
| `code/00_download.sh` | Downloads Paper Copilot, ICLR Insights, and PRISM files |
| `code/01_inspect.py` | Prints fields, coverage, and ID overlap of the raw files |
| `code/02_prepare.py` | Builds review- and paper-level tables for 2024 and 2025; 80/20 split (2025 split from Project 1) |
| `code/02b_prepare_2026.py` | Same for ICLR 2026 (different rating scale; used in the final report) |
| `code/10_llm_pilot.py` | Compares two LLMs on 200 reviews (agreement, repeatability, speed, cost) |
| `code/11_llm_extract.py` | Codes every review body: sentiment, main concern and whether it is fixable, specificity |
| `code/07_llm_check.py` | Checks the codes against the rating histories |
| `code/12_llm_style.py` | Marker-word index of LLM-like wording, by year |
| `code/03_decision.py` | Bayesian AC decision model (NumPyro NUTS); PSIS-LOO and test-set scores |
| `code/04_summarize.py` | Model-comparison and train-year by test-year tables |
| `code/05_simulate.py` | Simulation check: recovery of known parameters |
| `code/08_baselines.py` | Baselines similar to the public tools (mean logistic, percentile logistic, nearest twins) |
| `code/09_importance.py` | Importance of each part of the fitted rule, with posterior intervals |
| `code/06_figures.py` | Figures, including the posterior predictive check |

Model fits write small JSON summaries to `proj2/data/processed/decision/`; posterior samples,
logs, and data tables are not tracked.
