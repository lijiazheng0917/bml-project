"""Non-Bayesian baselines modelled on the two public tools, scored on the same test sets.

  knn_twins  iclr-ac-simulator style: each paper -> percentiles (within its year) of its mean,
             lowest and highest rating; distance sqrt(dmean^2 + 0.5 dlow^2 + 0.25 dhigh^2)
             + 0.02 |d n_reviews|; predicted P(accept) = accept share of the 150 nearest
             training papers (ties included).
  pct_logit  goodpapers style: logistic regression on the percentiles of mean, min, max,
             spread, and the confidences of the lowest and highest rater.
  mean_logit logistic regression on the mean rating only.

Usage (from proj2/):  python code/08_baselines.py  -> data/processed/decision/baselines.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def metrics(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {"n": int(len(y)), "elpd_per_paper": float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p)}


def features():
    rev = pd.read_parquet(PROC / "reviews.parquet", columns=["year", "paper_id", "r1", "c1"])
    pap = pd.read_parquet(PROC / "papers.parquet")
    pap = pap[pap.in_analysis & pap.split.isin(["train", "test"])]
    g = rev.groupby(["year", "paper_id"])
    f = pd.DataFrame({"mean": g.r1.mean(), "low": g.r1.min(), "high": g.r1.max(), "sd": g.r1.std(ddof=0),
                      "n": g.size()}).reset_index()
    lo_c = rev.loc[rev.groupby(["year", "paper_id"]).r1.idxmin(), ["year", "paper_id", "c1"]].rename(columns={"c1": "c_low"})
    hi_c = rev.loc[rev.groupby(["year", "paper_id"]).r1.idxmax(), ["year", "paper_id", "c1"]].rename(columns={"c1": "c_high"})
    f = f.merge(lo_c, on=["year", "paper_id"]).merge(hi_c, on=["year", "paper_id"])
    pap = pap.merge(f, on=["year", "paper_id"])
    # percentiles within year, over ALL reviews / papers of that year (as the tools do)
    allrev = rev.dropna(subset=["r1"])
    for y in (2024, 2025):
        ry = np.sort(allrev[allrev.year == y].r1.to_numpy())
        my = np.sort(f[f.year == y]["mean"].to_numpy())
        sy = np.sort(f[f.year == y]["sd"].to_numpy())
        pct = lambda arr, x: (np.searchsorted(arr, x, "left") + np.searchsorted(arr, x, "right")) / 2 / len(arr)
        m = pap.year == y
        pap.loc[m, "p_mean"] = pct(my, pap.loc[m, "mean"].to_numpy())
        pap.loc[m, "p_low"] = pct(ry, pap.loc[m, "low"].to_numpy())
        pap.loc[m, "p_high"] = pct(ry, pap.loc[m, "high"].to_numpy())
        pap.loc[m, "p_sd"] = pct(sy, pap.loc[m, "sd"].to_numpy())
    return pap


def knn(train, test, k=150):
    A = train[["p_mean", "p_low", "p_high", "n"]].to_numpy()
    B = test[["p_mean", "p_low", "p_high", "n"]].to_numpy()
    acc = train.accepted.to_numpy()
    out = np.empty(len(B))
    for i, b in enumerate(B):
        d = np.sqrt((A[:, 0] - b[0]) ** 2 + 0.5 * (A[:, 1] - b[1]) ** 2 + 0.25 * (A[:, 2] - b[2]) ** 2) + 0.02 * np.abs(A[:, 3] - b[3])
        cut = np.partition(d, k - 1)[k - 1] + 1e-9
        out[i] = acc[d <= cut].mean()
    return out


def main():
    pap = features()
    res = {}
    X_pct = ["p_mean", "p_low", "p_high", "p_sd", "c_low", "c_high"]
    for tr_years in ([2024, 2025], [2024], [2025]):
        tr = pap[pap.year.isin(tr_years) & (pap.split == "train")]
        tag = "+".join(map(str, tr_years))
        for name in ("mean_logit", "pct_logit", "knn_twins"):
            for y in (2024, 2025):
                te = pap[(pap.year == y) & (pap.split == "test")]
                if name == "knn_twins":
                    p = knn(tr, te)
                else:
                    cols = ["mean"] if name == "mean_logit" else X_pct
                    clf = LogisticRegression(C=1e3, max_iter=2000).fit(tr[cols], tr.accepted)
                    p = clf.predict_proba(te[cols])[:, 1]
                res[f"{name}|train{tag}|test{y}"] = metrics(te.accepted.to_numpy(), p)
    json.dump(res, open(PROC / "decision" / "baselines.json", "w"), indent=1)
    for k, v in res.items():
        print(f"{k:40s} elpd {v['elpd_per_paper']:.4f}  auc {v['auc']:.4f}  brier {v['brier']:.4f}")


if __name__ == "__main__":
    main()
