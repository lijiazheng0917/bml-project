"""Check the full LLM coding against known answers (initial vs final rating) and describe it.

Usage (from proj2/):  python code/07_llm_check.py   -> data/processed/llm_check.json
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
codes = pd.read_parquet(PROC / "llm_codes.parquet")
rev = pd.read_parquet(PROC / "reviews.parquet", columns=["year", "paper_id", "reviewer_id", "r0", "r1", "c0", "c1", "dr", "split", "in_analysis", "text"])
rev = rev[rev.in_analysis.fillna(False) & rev.text.notna()]
m = rev.merge(codes, on=["year", "paper_id", "reviewer_id"], how="left")
out = {"reviews_with_text": len(rev), "coded": int(m.body_sentiment.notna().sum()),
       "cost_usd": float(codes.cost.fillna(0).sum()), "median_latency_s": float(codes.latency.median())}
c = m.dropna(subset=["body_sentiment"])
for y in (2024, 2025):
    t = c[c.year == y]
    out[str(y)] = {
        "n": len(t),
        "rho_sentiment_initial": round(spearmanr(t.body_sentiment, t.r0)[0], 3),
        "rho_sentiment_final": round(spearmanr(t.body_sentiment, t.r1)[0], 3),
        "rho_specificity_conf": round(spearmanr(t.specificity, t.c1)[0], 3),
        "concern_type_share": t.main_concern_type.value_counts(normalize=True).round(3).to_dict(),
        "fixable_share": round(float(t.main_concern_fixable.mean()), 3),
        "update_any_raised": round(float((t[t.dr > 0]["update"] != "none").mean()), 3),
        "update_any_same": round(float((t[t.dr == 0]["update"] != "none").mean()), 3),
        "sentiment_mean_by_initial_rating": t.groupby("r0").body_sentiment.mean().round(2).to_dict(),
        "sentiment_sd_by_initial_rating": t.groupby("r0").body_sentiment.std().round(2).to_dict(),
    }
# among reviewers with the same initial rating, does a better-sounding text predict a raise?
t = c[c.r0.isin([3, 5, 6])]
out["raise_rate_by_fixable_given_r0"] = t.groupby(["r0", "main_concern_fixable"]).dr.apply(lambda s: round(float((s > 0).mean()), 3)).unstack().to_dict()
json.dump(out, open(PROC / "llm_check.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
