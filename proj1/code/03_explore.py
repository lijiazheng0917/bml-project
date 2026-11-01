"""Exploratory numbers and the three Project 1 figures (training split only).

Writes figures/fig{1,2,3}_*.{pdf,png} and data/processed/explore_summary.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PROC, FIG = ROOT / "data" / "processed", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Reference palette (light mode). Gray marks the comparison/baseline group.
BLUE, ORANGE = "#2a78d6", "#eb6834"
GRAY, GRAY_FILL = "#898781", "#c3c2b7"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})

papers = pd.read_parquet(PROC / "papers.parquet")
reviews = pd.read_parquet(PROC / "reviews.parquet")
P = papers[papers.in_analysis & (papers.split == "train")].copy()
R = reviews[reviews.in_analysis & (reviews.split == "train")].copy()
P["accepted"] = P["accepted"].astype(int)
P[["has_low1", "has_high1"]] = P[["has_low1", "has_high1"]].astype(bool)
P["stratum"] = P["mean1"].round(3)
out = {"n_train_papers": len(P), "n_train_reviews": len(R),
       "train_decisions": P["decision"].value_counts().to_dict(),
       "train_accept_rate": P["accepted"].mean()}


# ---------- helpers ----------
def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def binned_rate(df, edges, min_n=20):
    b = pd.cut(df["mean1"], edges, right=False)
    g = (df.groupby(b, observed=True)
           .agg(x=("mean1", "mean"), k=("accepted", "sum"), n=("accepted", "size"))
           .reset_index(drop=True))
    g = g[g.n >= min_n].copy()
    g["rate"] = g.k / g.n
    g["lo"], g["hi"] = wilson(g.k, g.n)
    return g


def stratified_diff(df, in_a, min_each=5, by="stratum"):
    """Risk difference P(accept | A) - P(accept | not A) within strata (default: equal
    final mean), Cochran-Mantel-Haenszel weights. Returns estimate, 95% CI, counts."""
    num = den = var = 0.0
    used = n_used = 0
    for _, s in df.groupby(by):
        a, b = s[in_a.loc[s.index]], s[~in_a.loc[s.index]]
        if len(a) < min_each or len(b) < min_each:
            continue
        ra, rb = a.accepted.mean(), b.accepted.mean()
        w = len(a) * len(b) / (len(a) + len(b))
        num += w * (ra - rb)
        den += w
        var += w ** 2 * (ra * (1 - ra) / len(a) + rb * (1 - rb) / len(b))
        used += 1
        n_used += len(s)
    est, se = num / den, np.sqrt(var) / den
    return {"diff_pp": 100 * est, "ci_pp": [100 * (est - 1.96 * se), 100 * (est + 1.96 * se)],
            "strata": used, "papers": n_used}


def rate_series(ax, g, color, label, marker):
    ax.vlines(g.x, 100 * g.lo, 100 * g.hi, color=color, lw=1.1, alpha=0.85, zorder=2)
    ax.plot(g.x, 100 * g.rate, color=color, lw=2, marker=marker, ms=6,
            mec="white", mew=1.2, label=label, zorder=3)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)


# ---------- descriptive numbers (Section 1.2 / 2.1) ----------
R["change"] = np.sign(R["dr"]).map({-1: "lowered", 0: "unchanged", 1: "raised"})
out["review_change_share"] = R["change"].value_counts(normalize=True).round(4).to_dict()
moves = R[R.dr != 0].groupby(["r0", "r1"]).size().sort_values(ascending=False)
out["top_moves"] = {f"{int(a)}->{int(b)}": int(n) for (a, b), n in moves.head(6).items()}
out["paper_traj"] = P["traj"].value_counts().to_dict()


def icc(df, col):
    d = df.dropna(subset=[col])
    g = d.groupby("paper_id")[col]
    k = g.size()
    keep = k[k >= 2].index
    d = d[d.paper_id.isin(keep)]
    g = d.groupby("paper_id")[col]
    within = (g.var(ddof=1) * (g.size() - 1)).sum() / (len(d) - g.ngroups)
    kbar = g.size().mean()
    between = max(g.mean().var(ddof=1) - within / kbar, 0)
    return between / (between + within), float(np.sqrt(within))


out["icc_initial"], out["within_sd_initial"] = icc(R, "r0")
out["icc_final"], out["within_sd_final"] = icc(R, "r1")
cols = {"r1": "rating", "snd1": "soundness", "prs1": "presentation", "ctb1": "contribution", "c1": "confidence"}
out["spearman_final"] = R[list(cols)].rename(columns=cols).corr(method="spearman").round(3).to_dict()
out["accept_rate_by_n_rev"] = P.groupby("n_rev")["accepted"].agg(["mean", "size"]).round(3).to_dict()
# selection: initial scores of papers withdrawn before a decision vs decided papers (all papers)
sel = papers[papers.in_insights & papers.decision.isin(["withdrawn"] + ["reject", "poster", "spotlight", "oral"])]
out["initial_mean_withdrawn_vs_decided"] = (sel.assign(grp=np.where(sel.decision == "withdrawn", "withdrawn", "decided"))
                                            .groupby("grp")["mean0"].agg(["mean", "median", "size"]).round(3).to_dict())
out["withdrawn_share_by_initial_mean"] = (sel.assign(wd=sel.decision.eq("withdrawn"),
                                                     b=pd.cut(sel.mean0, [0, 3, 4, 5, 6, 11], right=False))
                                          .groupby("b", observed=True)["wd"].agg(["mean", "size"]).round(3)
                                          .rename(index=str).to_dict())

# ---------- H1: low-score veto ----------
npos = R.assign(pos=R["r1"] >= 6).groupby("paper_id")["pos"].sum()
P["n_pos1"] = P["paper_id"].map(npos).astype(int)
P["share_pos1"] = P["n_pos1"] / P["n_rev"]
P["stratum_votes"] = list(zip(P["stratum"], P["n_rev"], P["n_pos1"]))
out["H1_low"] = stratified_diff(P, P["has_low1"])
out["H1_high"] = stratified_diff(P, P["has_high1"])
# same mean AND same number of reviewers on the accept side (rating >= 6)
out["H1_low_given_votes"] = stratified_diff(P, P["has_low1"], by="stratum_votes")
out["H1_high_given_votes"] = stratified_diff(P, P["has_high1"], by="stratum_votes")
# vote counting: majority of reviewers at >= 6, at the same mean
out["H1_majority_pos"] = stratified_diff(P, P["share_pos1"] > 0.5)
out["H1_unanimous_pos"] = stratified_diff(P, P["share_pos1"] == 1.0)
cells = P[P.mean1.between(5, 6.5)].groupby(["has_low1", "has_high1"]).size()
out["H1_cells_band"] = {f"low={a},high={b}": int(n) for (a, b), n in cells.items()}

edges = np.arange(3.0, 8.01, 0.5)

# Figure 1: small multiples. Each panel fixes the final mean (4-reviewer papers only,
# so the mean is exact); each dot is one combination of the four final ratings.
four = P[P.n_rev == 4].copy()
combo = (R[R.paper_id.isin(four.paper_id)].groupby("paper_id")["r1"]
           .apply(lambda x: tuple(sorted(int(v) for v in x))))
four["combo"] = four["paper_id"].map(combo)
cg = (four.groupby(["stratum", "combo"])
          .agg(n=("accepted", "size"), k=("accepted", "sum"),
               sd=("sd1", "first"), votes=("n_pos1", "first"))
          .reset_index())
cg = cg[cg.n >= 10].copy()
cg["rate"] = cg.k / cg.n
cg["lo"], cg["hi"] = wilson(cg.k, cg.n)
out["fig1_combos"] = [
    {"mean": float(r.stratum), "combo": list(r.combo), "sd": round(float(r.sd), 3),
     "votes_ge6": int(r.votes), "n": int(r.n), "accept_pct": round(100 * float(r.rate), 1)}
    for r in cg.sort_values(["stratum", "sd"]).itertuples()]
out["fig1_n_four_reviewer_papers"] = int(len(four))

VOTE_RAMP = {0: "#86b6ef", 1: "#5598e7", 2: "#2a78d6", 3: "#1c5cab", 4: "#0d366b"}
panel_means = [4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5]
fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.9), sharex=True, sharey=True)
for ax, m in zip(axes.flat, panel_means):
    s = cg[np.isclose(cg.stratum, m)].sort_values("sd").reset_index(drop=True)
    for i, r in s.iterrows():
        col = VOTE_RAMP[int(r.votes)]
        x, y = r.sd, 100 * r.rate
        ax.vlines(x, 100 * r.lo, 100 * r.hi, color=col, lw=1.1, alpha=0.85, zorder=2)
        ax.plot(x, y, "o", ms=6, color=col, mec="white", mew=1.1, zorder=3)
        # label to the right unless the next dot sits right there; then go left
        crowded = i + 1 < len(s) and s.sd[i + 1] - x < 1.0 and abs(100 * s.rate[i + 1] - y) < 9
        ax.annotate(",".join(map(str, r.combo)), (x, y), xytext=(-5 if crowded else 5, 0),
                    textcoords="offset points", fontsize=6.5, color=INK2, va="center",
                    ha="right" if crowded else "left", zorder=4)
    ax.set_title(f"Mean = {m:.2f}", fontsize=8.5, loc="left")
    ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(-4, 104)
fig.supxlabel("Spread of the four final ratings (standard deviation)", fontsize=8.5, color=INK2)
fig.supylabel("Accepted (%)", fontsize=8.5, color=INK2)
handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=c, mec="white", label=str(v))
           for v, c in VOTE_RAMP.items()]
fig.legend(handles=handles, title="Number of reviewers rating ≥ 6 (out of 4)",
           title_fontsize=8, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig1_spread")

# ---------- H2: raised scores ----------
sub = P[P.traj != "mixed_or_lowered"]
out["H2_raised_vs_unchanged"] = stratified_diff(sub, sub["traj"].eq("raised"))
out["H2_raised_vs_unchanged_given_votes"] = stratified_diff(sub, sub["traj"].eq("raised"),
                                                            by="stratum_votes")
# raises that moved a reviewer across the accept line (<6 -> >=6)
cross = R[(R.r0 < 6) & (R.r1 >= 6)].groupby("paper_id").size()
sub = sub.assign(crossed=sub["paper_id"].isin(cross.index))
cu = sub[sub.crossed | sub.traj.eq("unchanged")]
out["H2_crossed_vs_unchanged"] = stratified_diff(cu, cu["crossed"])
out["H2_crossed_vs_unchanged_given_votes"] = stratified_diff(cu, cu["crossed"], by="stratum_votes")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
rate_series(ax, binned_rate(sub[sub.traj == "unchanged"], edges), GRAY, "No change", "o")
rate_series(ax, binned_rate(sub[sub.traj == "raised"], edges), BLUE, "≥1 reviewer raised", "s")
ax.set_title("A  Acceptance at the same final mean")
ax.set_xlabel("Final mean rating (bins of 0.5)")
ax.set_ylabel("Accepted (%)")
ax.set_xlim(3.0, 8.0)
ax.set_ylim(-3, 103)
ax.legend(loc="upper left")

ax = axes[1]
tab = (R.groupby("r0")["change"].value_counts(normalize=True).unstack(fill_value=0)
         .reindex(columns=["lowered", "unchanged", "raised"], fill_value=0))
ns = R.groupby("r0").size()
ypos = np.arange(len(tab))
left = np.zeros(len(tab))
for col, color in [("lowered", ORANGE), ("unchanged", GRAY_FILL), ("raised", BLUE)]:
    w = 100 * tab[col].to_numpy()
    ax.barh(ypos, w, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.5,
            label=col.capitalize())
    if col == "raised":
        for y, (l0, wi) in enumerate(zip(left, w)):
            if wi >= 9:
                ax.text(l0 + wi / 2, y, f"{wi:.0f}%", ha="center", va="center", color="white", fontsize=7.5)
    left += w
ax.set_yticks(ypos, [f"{int(r)}  (n={ns[r]:,})" for r in tab.index])
ax.set_xlim(0, 100)
ax.set_xlabel("Share of reviewers (%)")
ax.set_ylabel("Initial rating")
ax.set_title("B  How reviewers changed after rebuttal")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.tight_layout()
save(fig, "fig2_raised")

# ---------- H3: confidence ----------
R["absdev0"] = (R["r0"] - R["others_r0"]).abs()
cal = R.dropna(subset=["absdev0", "c0"]).groupby("c0")["absdev0"].agg(["mean", "std", "size"])
cal["se"] = cal["std"] / np.sqrt(cal["size"])
out["calibration_absdev_by_conf"] = cal[["mean", "size"]].round(3).to_dict()
out["spearman_conf_absdev"] = float(R[["c0", "absdev0"]].corr(method="spearman").iloc[0, 1])

band = P[P.mean1.between(5.0, 6.5)].copy()
base = P.groupby("stratum")["accepted"].agg(["mean", "size"])
band = band.join(base, on="stratum")
band = band[band["size"] >= 10]
band["excess"] = band["accepted"] - band["mean"]
bins = [-np.inf, -0.3, -0.1, 0.1, 0.3, np.inf]
labels = ["< −0.3", "−0.3 to −0.1", "−0.1 to 0.1", "0.1 to 0.3", "> 0.3"]
band["dbin"] = pd.cut(band["dconf"], bins, labels=labels, include_lowest=True)
ex = band.groupby("dbin", observed=False)["excess"].agg(["mean", "std", "size"])
ex["se"] = ex["std"] / np.sqrt(ex["size"])
out["H3_excess_by_dconf"] = ex[["mean", "size"]].round(4).to_dict()
pos, neg = band["dconf"] >= 0.1, band["dconf"] <= -0.1
two = band[pos | neg]
out["H3_pos_vs_neg"] = stratified_diff(two, pos[pos | neg])
out["H3_pos_vs_neg_given_votes"] = stratified_diff(two, pos[pos | neg], by="stratum_votes")
out["H3_band_n"] = len(band)


def kfmt(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{int(n)}"

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax = axes[0]
ax.errorbar(cal.index, cal["mean"], yerr=1.96 * cal["se"], color=BLUE, lw=2, marker="o", ms=6,
            mec="white", mew=1.2, ecolor=BLUE, elinewidth=1.1, capsize=0)
ax.set_xticks(cal.index, [f"{int(c)}\nn={kfmt(n)}" for c, n in zip(cal.index, cal["size"])])
ax.set_xlabel("Reviewer's initial confidence")
ax.set_ylabel("|own − others' mean| (initial)")
ax.set_title("A  Distance from co-reviewers")

ax = axes[1]
xs = np.arange(len(ex))
ax.axhline(0, color=AXIS, lw=1, zorder=1)
ax.errorbar(xs, 100 * ex["mean"], yerr=1.96 * 100 * ex["se"], color=BLUE, lw=0, marker="o", ms=6,
            mec="white", mew=1.2, ecolor=BLUE, elinewidth=1.1, capsize=0, zorder=3)
ax.set_xticks(xs, [f"{l}\nn={kfmt(n)}" for l, n in zip(labels, ex["size"])], fontsize=7)
ax.set_xlabel("Confidence-weighted mean − plain mean")
ax.set_ylabel("Acceptance above expected (pp)")
ax.set_title("B  Papers with final mean 5.0–6.5")
fig.tight_layout()
save(fig, "fig3_confidence")

json.dump(out, open(PROC / "explore_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
