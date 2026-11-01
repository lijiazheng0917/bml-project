# Predicting ICLR 2025 peer-review decisions with Bayesian models

Course project for CSE 5825 Bayesian Machine Learning (UConn), Jiazheng Li.

The project asks whether the information in peer reviews predicts the decision on an ICLR 2025
submission (reject, poster, spotlight, oral), and in particular whether reviews carry information
beyond the mean rating. It uses public OpenReview-derived data for 11,672 submissions.

| Folder | Content |
|---|---|
| `proj0/` | Project 0: proposal (LaTeX + PDF) and slides |
| `proj1/` | Project 1: data exploration, figures, model design, hypotheses |

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

Runs on the UConn Storrs HPC through SLURM (adjust the account and paths for other machines).

The report is in `proj1/report/proj1_report.pdf`.

```bash
# Python environment, and Tectonic for the LaTeX report
mamba create -n bml -c conda-forge python=3.12 pandas pyarrow matplotlib-base numpy scipy
mamba create -n tex -c conda-forge tectonic

cd proj1
srun -p general -A <account> -t 00:20:00 bash code/00_download.sh              # raw data -> data/raw
srun -p general -A <account> -c2 --mem=8G -t 00:30:00 bash code/run_all.sh     # tables, figures, report
```

| Script | What it does |
|---|---|
| `code/00_download.sh` | Downloads the three raw files |
| `code/01_inspect.py` | Prints fields, label values and ID overlap of the raw files |
| `code/02_prepare.py` | Builds `reviews.parquet` and `papers.parquet`; fixes decision labels; 80/20 train/test split |
| `code/03_explore.py` | Figures 1-3 and summary numbers (training split only) |

The 20% test split is held out for the hypothesis tests in later project stages.
