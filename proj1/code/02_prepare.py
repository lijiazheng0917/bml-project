"""Build the review-level and paper-level tables used in Project 1.

Inputs  (data/raw):  ICLR Insights scores, ReviewerToo full release, Paper Copilot list.
Outputs (data/processed):
  reviews.parquet  one row per (paper, reviewer): initial/final rating, confidence, sub-scores
  papers.parquet   one row per paper: decision, area, score summaries, train/test split
  sample_counts.json  counts used for the sample-construction table
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
SEED = 5825
DECIDED = ["reject", "poster", "spotlight", "oral"]


def norm_label(s):
    """Map raw decision/status strings to one label set."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).lower()
    for key, lab in [("withdraw", "withdrawn"), ("desk", "desk_reject"), ("oral", "oral"),
                     ("spotlight", "spotlight"), ("poster", "poster"), ("reject", "reject")]:
        if key in s:
            return lab
    return None


def pair(s):
    """'5;6' -> (5.0, 6.0); anything unparseable -> (nan, nan)."""
    try:
        a, b = str(s).split(";")
        return float(a), float(b)
    except (ValueError, TypeError):
        return np.nan, np.nan


# ---------- reviewer-level scores (ICLR Insights) ----------
ins_raw = json.load(open(RAW / "iclr2025_insights.json"))
# A few paper IDs appear twice; keep the entry with the most reviews.
ins_by_id = {}
for p in ins_raw:
    if p["id"] not in ins_by_id or len(p["review"]) > len(ins_by_id[p["id"]]["review"]):
        ins_by_id[p["id"]] = p
ins = list(ins_by_id.values())
rows = []
for p in ins:
    for rid, r in p["review"].items():
        rec = {"paper_id": p["id"], "reviewer_id": rid, "tracing_score": p["tracing_score"]}
        for field, short in [("rating", "r"), ("confidence", "c"), ("soundness", "snd"),
                             ("presentation", "prs"), ("contribution", "ctb")]:
            rec[f"{short}0"], rec[f"{short}1"] = pair(r.get(field))
        rows.append(rec)
reviews = pd.DataFrame(rows)
reviews = reviews.dropna(subset=["r0", "r1"]).reset_index(drop=True)
reviews["dr"] = reviews["r1"] - reviews["r0"]

# leave-one-out mean of the OTHER reviewers' initial rating (for calibration checks)
g = reviews.groupby("paper_id")["r0"]
n_rev = g.transform("size")
reviews["others_r0"] = (g.transform("sum") - reviews["r0"]) / (n_rev - 1)
reviews.loc[n_rev < 2, "others_r0"] = np.nan

# ---------- paper-level labels ----------
rt = pd.read_parquet(RAW / "reviewertoo_iclr2025_full.parquet")
rt = rt[["paper_id", "title", "primary_area", "decision", "metareview", "citations_serper"]].copy()
rt["has_metareview"] = rt["metareview"].fillna("").str.len() > 0
rt["label_rt"] = rt["decision"].map(norm_label)
rt = rt.drop(columns=["metareview", "decision"])

pc_raw = pd.DataFrame(json.load(open(RAW / "papercopilot_iclr2025.json")))
pc_raw = pc_raw[["id", "status"]].rename(columns={"id": "paper_id", "status": "status_pc"})
dup_ids = pc_raw.loc[pc_raw["paper_id"].duplicated(keep=False)]
pc = pc_raw.drop_duplicates("paper_id", keep="first").copy()
pc["label_pc"] = pc["status_pc"].map(norm_label)

papers = rt.merge(pc, on="paper_id", how="outer")
# Decision rule:
#  - ReviewerToo folds withdrawals into "reject"; Paper Copilot marks them, so
#    reject + Withdraw + no meta-review -> withdrawn before any AC decision.
#  - reject + Withdraw + meta-review: the AC rejected it and the authors withdrew
#    afterwards; the AC decision exists, so keep "reject".
#  - accept + Withdraw (withdrawn after acceptance): the AC decision was accept, keep it.
#  - any other disagreement -> label_conflict, excluded from the analysis.
lr, lp = papers["label_rt"], papers["label_pc"]
papers["decision"] = lr
pre_decision_wd = (lr == "reject") & (lp == "withdrawn") & ~papers["has_metareview"]
papers.loc[pre_decision_wd, "decision"] = "withdrawn"
papers["withdrawn_after_reject"] = (lr == "reject") & (lp == "withdrawn") & papers["has_metareview"]
papers["withdrawn_after_accept"] = lr.isin(["poster", "spotlight", "oral"]) & (lp == "withdrawn")
conflict = (lr != lp) & ~((lr == "reject") & (lp == "withdrawn")) & ~papers["withdrawn_after_accept"]
papers.loc[conflict, "decision"] = "label_conflict"
papers["labels_agree"] = ~conflict

# ---------- paper-level score summaries ----------
def summarize(d):
    r0, r1, c1 = d["r0"].to_numpy(), d["r1"].to_numpy(), d["c1"].to_numpy()
    wmean1 = np.nansum(c1 * r1) / np.nansum(c1) if np.nansum(c1) > 0 else np.nan
    return pd.Series({
        "n_rev": len(d),
        "mean0": r0.mean(), "mean1": r1.mean(),
        "sd0": r0.std(ddof=1) if len(d) > 1 else np.nan,
        "sd1": r1.std(ddof=1) if len(d) > 1 else np.nan,
        "min1": r1.min(), "max1": r1.max(),
        "n_raised": int((r1 > r0).sum()), "n_lowered": int((r1 < r0).sum()),
        "conf_mean1": np.nanmean(c1),
        "wmean1": wmean1,
        "tracing_score": d["tracing_score"].iloc[0],
    })

summ = reviews.groupby("paper_id").apply(summarize, include_groups=False).reset_index()
summ["has_low1"] = summ["min1"] <= 3
summ["has_high1"] = summ["max1"] >= 8
summ["dconf"] = summ["wmean1"] - summ["mean1"]
summ["traj"] = np.select(
    [(summ["n_raised"] > 0) & (summ["n_lowered"] == 0),
     (summ["n_raised"] == 0) & (summ["n_lowered"] == 0)],
    ["raised", "unchanged"], default="mixed_or_lowered")

papers = papers.merge(summ, on="paper_id", how="left")
papers["in_insights"] = papers["n_rev"].notna()
for col in ("has_low1", "has_high1"):
    papers[col] = papers[col].astype("boolean")
papers["accepted"] = papers["decision"].map({"reject": 0, "poster": 1, "spotlight": 1, "oral": 1})
papers["in_analysis"] = papers["decision"].isin(DECIDED) & (papers["n_rev"] >= 2)

# ---------- train/test split (stratified by decision) ----------
rng = np.random.default_rng(SEED)
papers["split"] = "none"
for lab, idx in papers[papers["in_analysis"]].groupby("decision").groups.items():
    idx = np.array(sorted(idx))
    test = rng.choice(idx, size=int(round(0.2 * len(idx))), replace=False)
    papers.loc[idx, "split"] = "train"
    papers.loc[test, "split"] = "test"

reviews = reviews.merge(papers[["paper_id", "decision", "accepted", "split", "in_analysis"]],
                        on="paper_id", how="left")

# ---------- save ----------
reviews.to_parquet(PROC / "reviews.parquet", index=False)
papers.to_parquet(PROC / "papers.parquet", index=False)

counts = {
    "reviewertoo_papers": int(len(rt)),
    "papercopilot_rows": int(len(pc_raw)),
    "papercopilot_duplicate_rows": dup_ids.sort_values("paper_id").values.tolist(),
    "papercopilot_papers": int(len(pc)),
    "union_papers": int(len(papers)),
    "insights_entries": int(len(ins_raw)),
    "insights_papers": int(len(ins)),
    "withdrawn_after_accept": int(papers["withdrawn_after_accept"].sum()),
    "withdrawn_after_reject": int(papers["withdrawn_after_reject"].sum()),
    "insights_papers_with_reviews": int(reviews["paper_id"].nunique()),
    "insights_reviews": int(len(reviews)),
    "decision_all": papers["decision"].value_counts(dropna=False).to_dict(),
    "status_pc_raw": papers["status_pc"].value_counts(dropna=False).to_dict(),
    "label_disagreements": int((~papers["labels_agree"]).sum()),
    "label_disagreement_table": pd.crosstab(papers["label_rt"].fillna("NA"),
                                            papers["label_pc"].fillna("NA")).to_dict(),
    "withdrawn_with_metareview": int(((papers["decision"] == "withdrawn") & papers["has_metareview"]).sum()),
    "reject_without_metareview": int(((papers["decision"] == "reject") & ~papers["has_metareview"].fillna(False)).sum()),
    "decided_in_insights": papers[papers["decision"].isin(DECIDED)]["in_insights"].value_counts().to_dict(),
    "analysis_by_split_decision": pd.crosstab(papers.loc[papers.in_analysis, "decision"],
                                              papers.loc[papers.in_analysis, "split"]).to_dict(),
    "tracing_ge2_analysis": int((papers.in_analysis & (papers.tracing_score >= 2)).sum()),
}
json.dump(counts, open(PROC / "sample_counts.json", "w"), indent=1, default=str)
print(json.dumps(counts, indent=1, default=str))
