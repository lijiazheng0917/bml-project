"""How much each part of the AC rule moves acceptance probabilities (posterior importance).

For every posterior draw of the full model (text_train2425), switch one component off and
measure the mean absolute change in each paper's acceptance probability:
  votes       v(k) replaced by its least-squares straight line in k (same draw): "average the ratings"
  confidence  gamma = 0
  text        beta = 0
  initial     delta = 0
  area        area offsets = 0
Reported as percentage points, posterior mean and 95% interval, over all training papers and
over borderline papers (final mean 5.0 to 6.5). Also reports how many decisions would flip at a
50% cut-off (posterior mean).

Usage (from proj2/):  python code/09_importance.py  -> data/processed/decision/importance.json, figures/fig_importance.pdf
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("N_CHAINS", "4")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dec", HERE / "03_decision.py")
dec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dec)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = HERE.parent
DEC = ROOT / "data" / "processed" / "decision"


def probs(s, d, off=None, target=None):
    """Acceptance probability (draws x papers) with one component switched off."""
    v, g, b, de, tau = s["v"], s["gamma"], s["beta"], s["delta"], s["tau"]
    alpha = s["sigma_a"][:, None] * s["z_a"]
    lv = dec.LEVELS.astype(float)
    if off == "votes":  # straight line in k, fitted per draw by least squares
        X = np.stack([np.ones_like(lv), lv], 1)
        coef = np.linalg.lstsq(X, v.T, rcond=None)[0]          # (2, draws)
        v = (X @ coef).T
        v = v - v[:, [2]]
    if off == "confidence":
        g = np.zeros_like(g)
    if off == "text":
        b = np.zeros_like(b)
    if off == "initial":
        de = np.zeros_like(de)
    if off == "area":
        alpha = np.zeros_like(alpha)
    mask, r1, r0, c1, tx = d["mask"], d["r1"], d["r0"], d["c1"], d["tx"]
    out = np.empty((v.shape[0], mask.shape[0]), np.float32)
    for s0 in range(0, v.shape[0], 200):
        sl = slice(s0, s0 + 200)
        lw = g[sl, None, None] * c1[None] + np.where(mask, 0.0, -1e9)[None]
        w = np.exp(lw - lw.max(2, keepdims=True))
        w = w / w.sum(2, keepdims=True)
        vr1 = v[sl][:, r1]
        vr0 = v[sl][:, r0]
        S = np.sum(w * ((1 - de[sl, None, None]) * vr1 + de[sl, None, None] * vr0 + b[sl, None, None] * tx[None]), 2)
        eta = S - tau[sl][:, d["year"]] - alpha[sl][:, d["area"]]
        if target is not None:
            # keep each year's acceptance rate: shift the year threshold so the mean probability
            # matches the full model in that draw (the rule changes, the selectivity does not)
            for yi in np.unique(d["year"]):
                cols = d["year"] == yi
                lo = np.full(eta.shape[0], -30.0)
                hi = np.full(eta.shape[0], 30.0)
                for _ in range(50):
                    mid = (lo + hi) / 2
                    pm = (1 / (1 + np.exp(-(eta[:, cols] + mid[:, None])))).mean(1)
                    big = pm > target[sl][:, yi]
                    hi = np.where(big, mid, hi)
                    lo = np.where(big, lo, mid)
                eta[:, cols] += ((lo + hi) / 2)[:, None]
        out[sl] = 1 / (1 + np.exp(-eta))
    return out


NAMES = {"votes": "Votes (5 to 6 step)", "confidence": "Confidence weight", "text": "Text beyond rating",
         "area": "Research area", "initial": "Initial rating"}


def main():
    import sys
    if "--plot-only" in sys.argv:  # redraw from the saved results, no recomputation
        plot(json.load(open(DEC / "importance.json")))
        return
    s = dict(np.load(DEC / "samples_text_train2425.npz"))
    d, pap = dec.load([2024, 2025], "train", "text_z")
    # papers.parquet already holds the final mean rating (mean1)
    border = ((pap.mean1 >= 5.0) & (pap.mean1 <= 6.5)).to_numpy()
    P = probs(s, d)
    years = np.unique(d["year"])
    target = np.stack([P[:, d["year"] == yi].mean(1) for yi in range(int(years.max()) + 1)], 1)
    res = {"n_papers": int(len(pap)), "n_borderline": int(border.sum())}
    names = {"votes": "Votes (5 to 6 step)", "confidence": "Confidence weight", "text": "Text beyond rating",
             "area": "Research area", "initial": "Initial rating"}
    for k in names:
        Q = probs(s, d, off=k, target=target)
        diff = np.abs(P - Q) * 100
        flips = ((P > 0.5) != (Q > 0.5)).sum(1)
        res[k] = {
            "all_pp": [float(diff.mean(1).mean()), *np.percentile(diff.mean(1), [2.5, 97.5]).tolist()],
            "border_pp": [float(diff[:, border].mean(1).mean()), *np.percentile(diff[:, border].mean(1), [2.5, 97.5]).tolist()],
            "max_pp_mean": float(diff.mean(0).max()),
            "flips": [float(flips.mean()), *np.percentile(flips, [2.5, 97.5]).tolist()],
        }
        print(k, {kk: np.round(vv, 2).tolist() for kk, vv in res[k].items()}, flush=True)
    json.dump(res, open(DEC / "importance.json", "w"), indent=1)
    plot(res)


def plot(res):
    names = NAMES
    order = sorted(names, key=lambda k: res[k]["border_pp"][0])
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    y = np.arange(len(order))
    for off, key, col, lab in [(-0.15, "all_pp", "#8a8a8a", "all papers"), (0.15, "border_pp", "#1f6fd1", "borderline papers (mean 5 to 6.5)")]:
        mm = np.array([res[k][key][0] for k in order])
        lo = np.array([res[k][key][1] for k in order])
        hi = np.array([res[k][key][2] for k in order])
        lo = np.maximum(lo, 0.05)  # log axis: show lower bounds at or below 0.05 at the edge
        ax.errorbar(mm, y + off, xerr=[mm - lo, hi - mm], fmt="o", color=col, ms=4, capsize=2, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in order])
    ax.set_xlabel("Change in acceptance probability (pp, log scale)")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 30)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fig_importance.pdf")


if __name__ == "__main__":
    main()
