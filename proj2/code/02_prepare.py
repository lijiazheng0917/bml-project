"""Build the Project 2 tables for ICLR 2024 and 2025.

Inputs (data/raw): Paper Copilot lists, ICLR Insights rating histories, PRISM review text.
The 2025 train/test split is taken from Project 1 (../proj1/data/processed/papers.parquet)
so that the held-out 2025 test set stays exactly the one fixed in Project 1.

Outputs (data/processed):
  reviews.parquet  one row per (year, paper, reviewer): initial/final rating and confidence,
                   final sub-scores, reply counts (2024 only), review text (if matched)
  papers.parquet   one row per (year, paper): decision, area, summaries, split
  prepare_counts.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
P1 = ROOT.parent / "proj1" / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
SEED = 5825
DECIDED = ["reject", "poster", "spotlight", "oral"]
COUNTS = {}


def label(s):
    s = str(s or "").lower()
    for key, lab in [("withdraw", "withdrawn"), ("desk", "desk_reject"), ("oral", "oral"),
                     ("spotlight", "spotlight"), ("poster", "poster"), ("reject", "reject"),
                     ("pending", "pending")]:
        if key in s:
            return lab
    return None


def pair(s):
    try:
        a, b = str(s).split(";")
        return float(a), float(b)
    except (ValueError, TypeError):
        return np.nan, np.nan


def lead_num(s):
    m = re.match(r"\s*(\d+)", str(s or ""))
    return float(m.group(1)) if m else np.nan


def semi(s):
    return [x for x in str(s or "").split(";")]


def build_year(y):
    # ---- Paper Copilot: per-reviewer final values, aligned by the reviewers list ----
    pc = pd.DataFrame(json.load(open(RAW / f"papercopilot_iclr{y}.json")))
    pc = pc[pc.track == "main"].drop_duplicates("id")
    pc_rev = []
    for p in pc.itertuples():
        ids = semi(p.reviewers)
        cols = {k: semi(getattr(p, k, "")) for k in
                ("rating", "confidence", "soundness", "presentation", "contribution",
                 "reply_reviewers", "wc_reply_reviewers", "wc_review")}
        for i, rid in enumerate(ids):
            if not rid:
                continue
            rec = {"paper_id": p.id, "reviewer_id": rid}
            for k, v in cols.items():
                rec[f"pc_{k}"] = float(v[i]) if i < len(v) and v[i] not in ("", None) else np.nan
            pc_rev.append(rec)
    pc_rev = pd.DataFrame(pc_rev)

    # ---- ICLR Insights: initial and final rating and confidence ----
    ins_by_id = {}
    for p in json.load(open(RAW / f"iclr{y}_insights.json")):
        if p["id"] not in ins_by_id or len(p["review"]) > len(ins_by_id[p["id"]]["review"]):
            ins_by_id[p["id"]] = p
    rows = []
    for p in ins_by_id.values():
        for rid, r in p["review"].items():
            rec = {"paper_id": p["id"], "reviewer_id": rid, "tracing_score": p["tracing_score"]}
            rec["r0"], rec["r1"] = pair(r.get("rating"))
            rec["c0"], rec["c1"] = pair(r.get("confidence"))
            rows.append(rec)
    rev = pd.DataFrame(rows).dropna(subset=["r0", "r1"])
    rev = rev.merge(pc_rev, on=["paper_id", "reviewer_id"], how="left")
    COUNTS[f"{y}_insights_reviews"] = len(rev)
    COUNTS[f"{y}_insights_reviews_matched_to_pc"] = int(rev.pc_rating.notna().sum())
    COUNTS[f"{y}_final_rating_agrees_with_pc"] = float((rev.r1 == rev.pc_rating)[rev.pc_rating.notna()].mean())
    rev = rev.rename(columns={"pc_soundness": "snd1", "pc_presentation": "prs1", "pc_contribution": "ctb1",
                              "pc_reply_reviewers": "n_reply", "pc_wc_reply_reviewers": "wc_reply",
                              "pc_wc_review": "wc_review"})
    if y != 2024:  # reply counts are only public for 2024
        rev[["n_reply", "wc_reply"]] = np.nan
    rev["year"] = y
    rev["dr"] = rev.r1 - rev.r0

    # ---- PRISM text, matched within paper on final (confidence, sub-scores[, rating]) ----
    pr = pd.read_parquet(RAW / "prism_papers.parquet",
                         columns=["paper_id", "venue", "decision", "meta_review", "reviews_json"])
    pr = pr[pr.venue == f"ICLR_{y}"]
    key = ["c1", "snd1", "prs1", "ctb1"] + (["r1"] if y != 2024 else [])
    texts = []
    by_paper = {pid: g for pid, g in rev.groupby("paper_id")}
    for pid, js in zip(pr.paper_id, pr.reviews_json):
        if pid not in by_paper or not js:
            continue
        g = by_paper[pid]
        gk = g[key].astype(float).apply(tuple, axis=1)
        prs = json.loads(js)
        pk = [tuple(lead_num(r.get(f)) for f in ["Confidence", "Soundness", "Presentation", "Contribution"]
                    + (["Rating"] if y != 2024 else [])) for r in prs]
        for r, k in zip(prs, pk):
            if pk.count(k) != 1:
                continue
            hit = g[gk == k]
            if len(hit) == 1:
                parts = [f"## {f}\n{str(r.get(f) or '').strip()}" for f in
                         ("Summary", "Strengths", "Weaknesses", "Questions", "Limitations") if str(r.get(f) or "").strip()]
                texts.append({"paper_id": pid, "reviewer_id": hit.reviewer_id.iloc[0],
                              "review_note_id": r.get("Review ID"), "text": "\n\n".join(parts)})
    texts = pd.DataFrame(texts)
    rev = rev.merge(texts, on=["paper_id", "reviewer_id"], how="left")
    COUNTS[f"{y}_reviews_with_text"] = int(rev.text.notna().sum())

    # ---- paper-level decision ----
    pap = pc[["id", "status", "primary_area", "title"]].rename(columns={"id": "paper_id"})
    pap["label_pc"] = pap.status.map(label)
    pm = pr[["paper_id", "decision", "meta_review"]].rename(columns={"decision": "decision_prism"})
    pm["label_prism"] = pm.decision_prism.map(label)
    pm["has_metareview"] = pm.meta_review.fillna("").str.len() > 20
    pap = pap.merge(pm.drop(columns="meta_review"), on="paper_id", how="left")
    pap["decision"] = pap.label_pc
    # Withdrawn after an AC decision: keep the AC decision (as in Project 1).
    wd = pap.label_pc == "withdrawn"
    after = wd & pap.has_metareview.fillna(False) & pap.label_prism.isin(DECIDED)
    pap.loc[after, "decision"] = pap.loc[after, "label_prism"]
    pap["withdrawn_after_decision"] = after
    conflict = pap.label_prism.isin(DECIDED) & pap.label_pc.isin(DECIDED) & (pap.label_prism != pap.label_pc)
    pap.loc[conflict, "decision"] = "label_conflict"
    COUNTS[f"{y}_decisions"] = pap.decision.value_counts(dropna=False).to_dict()
    COUNTS[f"{y}_label_conflicts"] = int(conflict.sum())

    g = rev.groupby("paper_id")
    summ = pd.DataFrame({
        "n_rev": g.size(), "mean0": g.r0.mean(), "mean1": g.r1.mean(),
        "n_ge6": g.r1.apply(lambda s: int((s >= 6).sum())), "n_ge8": g.r1.apply(lambda s: int((s >= 8).sum())),
        "n_raised": g.dr.apply(lambda s: int((s > 0).sum())), "tracing_score": g.tracing_score.first(),
        "n_text": g.text.apply(lambda s: int(s.notna().sum())),
    }).reset_index()
    pap = pap.merge(summ, on="paper_id", how="left")
    pap["year"] = y
    pap["accepted"] = pap.decision.map({"reject": 0, "poster": 1, "spotlight": 1, "oral": 1})
    pap["in_analysis"] = pap.decision.isin(DECIDED) & (pap.n_rev >= 2)
    return rev, pap


def split_2024(pap):
    rng = np.random.default_rng(SEED)
    pap = pap.reset_index(drop=True)
    pap["split"] = "none"
    for _, idx in pap[pap.in_analysis].groupby("decision").groups.items():
        idx = np.array(sorted(idx))
        test = rng.choice(idx, size=int(round(0.2 * len(idx))), replace=False)
        pap.loc[idx, "split"] = "train"
        pap.loc[test, "split"] = "test"
    return pap


def split_2025(pap):
    p1 = pd.read_parquet(P1 / "papers.parquet", columns=["paper_id", "split", "decision"])
    pap = pap.merge(p1.rename(columns={"decision": "decision_p1"}), on="paper_id", how="left")
    pap["split"] = pap.split.fillna("none")
    # Keep Project 1's sample definition exactly for 2025.
    COUNTS["2025_decision_differs_from_proj1"] = int(((pap.decision != pap.decision_p1) & pap.decision_p1.isin(DECIDED)).sum())
    pap["in_analysis"] = pap.split.isin(["train", "test"])
    pap["decision"] = np.where(pap.in_analysis, pap.decision_p1, pap.decision)
    pap["accepted"] = pap.decision.map({"reject": 0, "poster": 1, "spotlight": 1, "oral": 1})
    return pap.drop(columns="decision_p1")


revs, paps = [], []
for y in (2024, 2025):
    rev, pap = build_year(y)
    pap = split_2024(pap) if y == 2024 else split_2025(pap)
    rev = rev.merge(pap[["paper_id", "decision", "accepted", "split", "in_analysis", "primary_area"]],
                    on="paper_id", how="left")
    COUNTS[f"{y}_analysis_by_split_decision"] = pd.crosstab(pap.loc[pap.in_analysis, "decision"],
                                                            pap.loc[pap.in_analysis, "split"]).to_dict()
    a = rev[rev.in_analysis.fillna(False)]
    COUNTS[f"{y}_analysis_reviews"] = len(a)
    COUNTS[f"{y}_analysis_reviews_with_text"] = int(a.text.notna().sum())
    COUNTS[f"{y}_analysis_papers_all_text"] = int((pap.in_analysis & (pap.n_text == pap.n_rev)).sum())
    revs.append(rev)
    paps.append(pap)

reviews = pd.concat(revs, ignore_index=True)
papers = pd.concat(paps, ignore_index=True)
reviews.to_parquet(PROC / "reviews.parquet", index=False)
papers.to_parquet(PROC / "papers.parquet", index=False)
json.dump(COUNTS, open(PROC / "prepare_counts.json", "w"), indent=1, default=str)
print(json.dumps(COUNTS, indent=1, default=str))
