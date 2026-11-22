"""Figures for the Project 2 report (written to figures/).

  fig_v.pdf        how much each rating counts: posterior of v(k), free model vs linear-mean model,
                   fitted on 2024 alone and 2025 alone (stability across years)
  fig_params.pdf   gamma (confidence weight), delta (initial-rating weight), beta (text) with 95% intervals,
                   by training data
  fig_ppc.pdf      posterior predictive check on the TRAINING papers: observed versus simulated
                   acceptance rate for each combination of four ratings at equal mean (Project 1 Fig 1A)
  fig_calib.pdf    calibration on the test sets for the mean-only and full models

Usage (from proj2/):  python code/06_figures.py
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
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "figure.dpi": 150})
C = {"2024": "#1f6fd1", "2025": "#d1661f", "both": "#333333", "mean": "#8a8a8a"}


def load_samples(tag):
    f = DEC / f"samples_{tag}.npz"
    return dict(np.load(f)) if f.exists() else None


def interval(x, axis=0):
    return np.mean(x, axis), np.percentile(x, 2.5, axis), np.percentile(x, 97.5, axis)


def fig_v():
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    lv = dec.LEVELS
    for tag, lab, col, off in [("free_train24", "free v(k), 2024", C["2024"], -0.12),
                               ("free_train25", "free v(k), 2025", C["2025"], 0.12)]:
        s = load_samples(tag)
        if s is None:
            continue
        m, lo, hi = interval(s["v"])
        ax.errorbar(lv + off, m, yerr=[m - lo, hi - m], fmt="o-", color=col, ms=4, lw=1.4, capsize=2, label=lab)
    s = load_samples("mean_train2425")
    if s is not None:
        m, lo, hi = interval(s["v"])
        ax.plot(lv, m, "--", color=C["mean"], lw=1.2, label="linear (mean rating), both years")
    ax.set_xticks(lv)
    ax.set_xlabel("Final rating k")
    ax.set_ylabel("Value v(k) toward acceptance (v(5) = 0)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v.pdf")
    plt.close(fig)


def fig_params():
    rows = []
    for tag, lab in [("conf_train24", "2024"), ("conf_train25", "2025"), ("conf_train2425", "both"),
                     ("text_train24", "2024"), ("text_train25", "2025"), ("text_train2425", "both")]:
        s = load_samples(tag)
        if s is None:
            continue
        for p in ("gamma", "delta", "beta"):
            if p in s:
                m, lo, hi = interval(s[p])
                rows.append((p, lab, tag.split("_")[0], m, lo, hi))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["param", "data", "model", "m", "lo", "hi"])
    # one estimate per parameter and data: prefer the text model where it exists
    df = df.sort_values("model", ascending=False).drop_duplicates(["param", "data"])
    params = [p for p in ("gamma", "delta", "beta") if p in set(df.param)]
    names = {"gamma": "γ  confidence weight", "delta": "δ  weight on initial rating", "beta": "β  text beyond the rating"}
    fig, axes = plt.subplots(1, len(params), figsize=(2.3 * len(params), 2.4), sharey=True)
    axes = np.atleast_1d(axes)
    order = ["2024", "2025", "both"]
    for ax, p in zip(axes, params):
        sub = df[df.param == p].set_index("data").reindex(order).dropna()
        y = np.arange(len(sub))[::-1]
        ax.errorbar(sub.m, y, xerr=[sub.m - sub.lo, sub.hi - sub.m], fmt="o", color="#1f6fd1", ms=4, capsize=2)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"fit: {i}" for i in sub.index])
        ax.set_title(names[p], fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_params.pdf")
    plt.close(fig)


def fig_ppc(tag="conf_train2425", variant="conf"):
    s = load_samples(tag)
    if s is None:
        return
    d, pap = dec.load([2024, 2025], "train")
    P = dec.predict_p(s, d, variant)                  # (draws, papers)
    rev = pd.read_parquet(ROOT / "data" / "processed" / "reviews.parquet")
    combo = rev.groupby(["year", "paper_id"]).r1.apply(lambda x: ",".join(str(int(v)) for v in sorted(x)))
    pap = pap.merge(combo.rename("combo").reset_index(), on=["year", "paper_id"], how="left")
    pap["n"] = pap.combo.str.count(",") + 1
    four = pap[pap.n == 4].copy()
    four["mean"] = four.combo.apply(lambda c: np.mean([int(x) for x in c.split(",")]))
    four["n_ge6"] = four.combo.apply(lambda c: sum(int(x) >= 6 for x in c.split(",")))
    idx = four.index.to_numpy()
    rng = np.random.default_rng(0)
    draws = rng.choice(P.shape[0], 400, replace=False)
    sim = rng.random((400, len(idx))) < P[draws][:, idx]   # simulated decisions
    four["obs"] = four.accepted
    g = four.groupby("combo")
    keep = [c for c, x in g if len(x) >= 15 and 4.5 <= x["mean"].iloc[0] <= 6.5]
    rows = []
    for c in keep:
        loc = np.where(four.combo.to_numpy() == c)[0]
        rate = sim[:, loc].mean(1)
        rows.append((c, four["mean"].iloc[loc[0]], four.n_ge6.iloc[loc[0]], len(loc),
                     four.obs.iloc[loc].mean(), rate.mean(), *np.percentile(rate, [2.5, 97.5])))
    t = pd.DataFrame(rows, columns=["combo", "mean", "n_ge6", "n", "obs", "sim", "lo", "hi"]).sort_values(["mean", "n_ge6"])
    t.to_csv(DEC / "ppc_table.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    cmap = plt.get_cmap("Blues")
    for _, r in t.iterrows():
        col = cmap(0.35 + 0.15 * r.n_ge6)
        ax.plot([r.lo * 100, r.hi * 100], [r.obs * 100] * 2, color=col, lw=1.2)
        ax.plot(r.sim * 100, r.obs * 100, "o", color=col, ms=4)
    ax.plot([0, 100], [0, 100], color="#999", lw=0.8)
    ax.set_xlabel("Simulated acceptance rate (%), 95% interval")
    ax.set_ylabel("Observed acceptance rate (%)")
    for _, r in t[(t.n_ge6.isin([1, 3])) & (t["mean"].isin([5.25]))].iterrows():
        ax.annotate(r.combo, (r.sim * 100, r.obs * 100), fontsize=7, xytext=(4, -8), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ppc.pdf")
    plt.close(fig)
    covered = ((t.obs >= t.lo) & (t.obs <= t.hi)).mean()
    json.dump({"n_combos": len(t), "share_obs_inside_95": float(covered)}, open(DEC / "ppc_summary.json", "w"))


def fig_calib():
    summ = json.load(open(DEC / "summary.json")) if (DEC / "summary.json").exists() else []
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.8), sharey=True)
    for ax, y in zip(axes, (2024, 2025)):
        for tag, col, lab in [("mean_train2425", C["mean"], "mean only"), ("conf_train2425", C["both"], "free v, γ, δ"),
                              ("text_train2425", "#1f6fd1", "+ text")]:
            s = load_samples(tag)
            if s is None:
                continue
            variant = tag.split("_")[0]
            d, _ = dec.load([y], "test", "text_z" if variant == "text" else None)
            cal = dec.heldout(s, d, variant)["calibration"]
            p, o, n = np.array(cal).T
            ax.plot(p * 100, o * 100, "o-", color=col, ms=3, lw=1, label=lab)
        ax.plot([0, 100], [0, 100], color="#bbb", lw=0.8)
        ax.set_title(f"Test set {y}", fontsize=9)
        ax.set_xlabel("Predicted (%)")
    axes[0].set_ylabel("Observed acceptance (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_calib.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_v()
    fig_params()
    fig_ppc()
    fig_calib()
    print("figures:", sorted(os.listdir(FIG)))
