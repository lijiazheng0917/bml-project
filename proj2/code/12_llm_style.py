"""How LLM-like is each review? A marker-word index, by year (ICLR 2024, 2025, 2026).

Words whose frequency in peer reviews rose sharply after ChatGPT (adjectives and verbs typical of
LLM text). Index = marker words per 1,000 words of review body. Also reports, by year and by
index level, the rank correlation between the LLM-coded body sentiment and the rating, which
tells whether LLM-like text tracks the score more or less closely.

Usage (from proj2/):  python code/12_llm_style.py  -> data/processed/llm_style.json, data/processed/llm_style.parquet
"""
import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
MARKERS = ["commendable", "meticulous", "meticulously", "intricate", "intricacies", "notably", "noteworthy",
           "pivotal", "invaluable", "delve", "delves", "delving", "showcase", "showcases", "showcasing",
           "underscore", "underscores", "underscoring", "seamless", "seamlessly", "realm", "tapestry",
           "garnered", "commendably", "adeptly", "laudable", "elucidate", "elucidates", "holistic",
           "multifaceted", "nuanced", "furthermore", "additionally", "overall,"]
WORD = re.compile(r"[a-z]+,?")
MSET = set(MARKERS)


def index(text):
    words = WORD.findall(str(text).lower())
    n = len(words)
    hits = sum(1 for w in words if w in MSET or w.rstrip(",") in MSET)
    return n, hits


def load(suffix, rating_col):
    rev = pd.read_parquet(PROC / f"reviews{suffix}.parquet", columns=["year", "paper_id", "reviewer_id", "text", "in_analysis", "r1"] + (["r0"] if rating_col == "r0" else []))
    rev = rev[rev.in_analysis.fillna(False) & rev.text.notna()]
    codes = pd.read_parquet(PROC / f"llm_codes{suffix}.parquet", columns=["year", "paper_id", "reviewer_id", "body_sentiment"])
    return rev.merge(codes, on=["year", "paper_id", "reviewer_id"], how="left")


def main():
    df = pd.concat([load("", "r0"), load("_2026", "r1")], ignore_index=True)
    nh = df.text.map(index)
    df["n_words"] = [a for a, _ in nh]
    df["n_markers"] = [b for _, b in nh]
    df["llm_index"] = 1000 * df.n_markers / df.n_words.clip(lower=1)
    df["rating_for_sent"] = df.r0.fillna(df.r1)  # body sentiment vs initial rating (2026: final only)
    out = {}
    for y, g in df.groupby("year"):
        q = g.llm_index
        res = {"reviews": len(g), "median_words": float(g.n_words.median()),
               "markers_per_1000_words_mean": round(float(q.mean()), 3),
               "share_with_any_marker": round(float((g.n_markers > 0).mean()), 3),
               "share_index_above_2": round(float((q > 2).mean()), 3)}
        for w in ["commendable", "meticulous", "delve", "notably", "showcase", "underscore", "intricate", "pivotal"]:
            res[f"per_10k_words_{w}"] = round(float(g.text.str.lower().str.count(rf"\b{w}").sum() / g.n_words.sum() * 1e4), 3)
        c = g.dropna(subset=["body_sentiment"])
        res["rho_sent_rating_all"] = round(float(spearmanr(c.body_sentiment, c.rating_for_sent)[0]), 3)
        bins = pd.cut(c.llm_index, [-0.01, 0.0, 2.0, 1e9], labels=["none", "low (0-2]", "high (>2)"])
        res["by_index"] = {str(k): {"n": int(len(s)), "rho_sent_rating": round(float(spearmanr(s.body_sentiment, s.rating_for_sent)[0]), 3),
                                    "mean_sentiment": round(float(s.body_sentiment.mean()), 2), "mean_rating": round(float(s.rating_for_sent.mean()), 2)}
                           for k, s in c.groupby(bins, observed=True)}
        out[int(y)] = res
    json.dump(out, open(PROC / "llm_style.json", "w"), indent=1)
    df[["year", "paper_id", "reviewer_id", "n_words", "n_markers", "llm_index"]].to_parquet(PROC / "llm_style.parquet", index=False)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
