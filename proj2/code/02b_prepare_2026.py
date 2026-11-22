"""Build ICLR 2026 tables (separate from the 2024+2025 tables; different rating scale).

2026: ratings in {0,2,4,6,8,10}, no spotlight tier, no rating histories (final ratings only),
and the November 2025 OpenReview incident (reviews frozen, new ACs). Used for a separate
2026-only decision model.

Inputs (data/raw): Paper Copilot iclr2026.json, PRISM parquet.
Outputs (data/processed): reviews_2026.parquet, papers_2026.parquet, prepare_2026_counts.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
SEED = 5825
DECIDED = ["reject", "poster", "oral"]
C = {}


def label(s):
    s = str(s or "").lower()
    for key, lab in [("withdraw", "withdrawn"), ("desk", "desk_reject"), ("oral", "oral"),
                     ("spotlight", "spotlight"), ("poster", "poster"), ("reject", "reject"),
                     ("pending", "pending")]:
        if key in s:
            return lab
    return None


def lead_num(s):
    m = re.match(r"\s*(\d+)", str(s or ""))
    return float(m.group(1)) if m else np.nan


pc = pd.DataFrame(json.load(open(RAW / "papercopilot_iclr2026.json")))
pc = pc[pc.track == "main"].drop_duplicates("id")
C["main_track"] = len(pc)
rows = []
for p in pc.itertuples():
    ids = str(p.reviewers or "").split(";")
    cols = {k: str(getattr(p, k, "") or "").split(";") for k in
            ("rating", "confidence", "soundness", "presentation", "contribution", "wc_review")}
    for i, rid in enumerate(ids):
        if not rid:
            continue
        rec = {"paper_id": p.id, "reviewer_id": rid}
        for k, v in cols.items():
            rec[k] = float(v[i]) if i < len(v) and v[i] not in ("", None) else np.nan
        rows.append(rec)
rev = pd.DataFrame(rows).rename(columns={"rating": "r1", "confidence": "c1", "soundness": "snd1",
                                         "presentation": "prs1", "contribution": "ctb1"})
rev = rev.dropna(subset=["r1"])
rev["year"] = 2026

pr = pd.read_parquet(RAW / "prism_papers.parquet", columns=["paper_id", "venue", "decision", "meta_review", "reviews_json"])
pr = pr[pr.venue == "ICLR_2026"]
key = ["r1", "c1", "snd1", "prs1", "ctb1"]
texts = []
by_paper = {pid: g for pid, g in rev.groupby("paper_id")}
for pid, js in zip(pr.paper_id, pr.reviews_json):
    if pid not in by_paper or not js:
        continue
    g = by_paper[pid]
    gk = g[key].astype(float).apply(tuple, axis=1)
    prs = json.loads(js)
    pk = [tuple(lead_num(r.get(f)) for f in ["Rating", "Confidence", "Soundness", "Presentation", "Contribution"]) for r in prs]
    for r, k in zip(prs, pk):
        if pk.count(k) != 1:
            continue
        hit = g[gk == k]
        if len(hit) == 1:
            parts = [f"## {f}\n{str(r.get(f) or '').strip()}" for f in
                     ("Summary", "Strengths", "Weaknesses", "Questions", "Limitations") if str(r.get(f) or "").strip()]
            texts.append({"paper_id": pid, "reviewer_id": hit.reviewer_id.iloc[0], "text": "\n\n".join(parts)})
rev = rev.merge(pd.DataFrame(texts), on=["paper_id", "reviewer_id"], how="left")
C["reviews"] = len(rev)
C["reviews_with_text"] = int(rev.text.notna().sum())

pap = pc[["id", "status", "primary_area"]].rename(columns={"id": "paper_id"})
pap["label_pc"] = pap.status.map(label)
pm = pr[["paper_id", "decision", "meta_review"]].rename(columns={"decision": "decision_prism"})
pm["label_prism"] = pm.decision_prism.map(label)
pm["has_metareview"] = pm.meta_review.fillna("").str.len() > 20
pap = pap.merge(pm.drop(columns="meta_review"), on="paper_id", how="left")
pap["decision"] = pap.label_pc
after = (pap.label_pc == "withdrawn") & pap.has_metareview.fillna(False) & pap.label_prism.isin(DECIDED)
pap.loc[after, "decision"] = pap.loc[after, "label_prism"]
conflict = pap.label_prism.isin(DECIDED) & pap.label_pc.isin(DECIDED) & (pap.label_prism != pap.label_pc)
pap.loc[conflict, "decision"] = "label_conflict"
C["decisions"] = pap.decision.value_counts(dropna=False).to_dict()
C["label_conflicts"] = int(conflict.sum())
n = rev.groupby("paper_id").size().rename("n_rev")
pap = pap.merge(n, on="paper_id", how="left")
pap["year"] = 2026
pap["accepted"] = pap.decision.map({"reject": 0, "poster": 1, "oral": 1})
pap["in_analysis"] = pap.decision.isin(DECIDED) & (pap.n_rev >= 2)

rng = np.random.default_rng(SEED)
pap = pap.reset_index(drop=True)
pap["split"] = "none"
for _, idx in pap[pap.in_analysis].groupby("decision").groups.items():
    idx = np.array(sorted(idx))
    test = rng.choice(idx, size=int(round(0.2 * len(idx))), replace=False)
    pap.loc[idx, "split"] = "train"
    pap.loc[test, "split"] = "test"
rev = rev.merge(pap[["paper_id", "decision", "accepted", "split", "in_analysis", "primary_area"]], on="paper_id", how="left")
rev["r0"] = rev.r1  # no rating histories in 2026
C["analysis_by_split_decision"] = pd.crosstab(pap.loc[pap.in_analysis, "decision"], pap.loc[pap.in_analysis, "split"]).to_dict()
a = rev[rev.in_analysis.fillna(False)]
C["analysis_reviews"] = len(a)
C["analysis_reviews_with_text"] = int(a.text.notna().sum())
C["rating_values"] = a.r1.value_counts().sort_index().to_dict()
rev.to_parquet(PROC / "reviews_2026.parquet", index=False)
pap.to_parquet(PROC / "papers_2026.parquet", index=False)
json.dump(C, open(PROC / "prepare_2026_counts.json", "w"), indent=1, default=str)
print(json.dumps(C, indent=1, default=str))
