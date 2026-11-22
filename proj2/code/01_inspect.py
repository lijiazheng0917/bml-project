"""Inspect the Project 2 raw data: fields, coverage, and ID overlap across sources.

Run from proj2/:  python code/01_inspect.py
Prints a report; writes nothing.
"""
import collections
import json

import pandas as pd
import pyarrow.parquet as pq

RAW = "data/raw"
YEARS = (2024, 2025, 2026)


def split(v):
    """Paper Copilot stores per-reviewer values as 'a;b;c'."""
    if v is None or v == "":
        return []
    return [x for x in str(v).split(";") if x != ""]


# ---------- Paper Copilot ----------
pc = {}
for y in YEARS:
    d = json.load(open(f"{RAW}/papercopilot_iclr{y}.json"))
    main = [p for p in d if p.get("track") == "main"]
    st = collections.Counter(p.get("status") for p in main)
    rated = [p for p in main if split(p.get("rating"))]
    with_conf = [p for p in rated if len(split(p.get("confidence"))) == len(split(p.get("rating")))]
    ratings = collections.Counter(int(float(r)) for p in rated for r in split(p["rating"]))
    print(f"\n=== Paper Copilot ICLR {y}: {len(d)} rows, {len(main)} main track")
    print("  status:", dict(st.most_common()))
    print(f"  with ratings: {len(rated)}, with matching confidence: {len(with_conf)}")
    print("  rating values:", dict(sorted(ratings.items())))
    print("  reviews per paper:", dict(sorted(collections.Counter(len(split(p['rating'])) for p in rated).items())))
    for k in ("reply_reviewers", "wc_reply_authors"):
        nz = sum(1 for p in rated if any(x not in ("0", "") for x in split(p.get(k))))
        print(f"  papers with nonzero {k}: {nz}")
    pc[y] = {p["id"]: p for p in main}

# ---------- ICLR Insights ----------
for y in (2024, 2025):
    ins = json.load(open(f"{RAW}/iclr{y}_insights.json"))
    ids = {p["id"] for p in ins}
    changed = sum(1 for p in ins for r in p["review"].values()
                  if ";" in str(r.get("rating", "")) and len(set(str(r["rating"]).split(";"))) > 1)
    print(f"\n=== Insights {y}: {len(ins)} papers, overlap with Paper Copilot main: {len(ids & set(pc[y]))}")
    print("  review fields:", sorted(next(iter(ins[1]["review"].values())).keys()))
    print("  reviewer records whose rating changed:", changed)

# ---------- PRISM ----------
f = pq.ParquetFile(f"{RAW}/prism_papers.parquet")
print(f"\n=== PRISM parquet: {f.metadata.num_rows} rows")
print("  columns:", f.schema_arrow.names)
cols = ["paper_id", "venue", "decision", "num_reviews", "reviews_json", "meta_review", "full_text"]
df = pd.read_parquet(f"{RAW}/prism_papers.parquet", columns=cols)
df = df[df.venue.str.startswith("ICLR")]
for y in YEARS:
    s = df[df.venue == f"ICLR_{y}"]
    ids = set(s.paper_id)
    print(f"\n--- PRISM ICLR {y}: {len(s)} papers; overlap with Paper Copilot main: {len(ids & set(pc[y]))}")
    print("  decision:", dict(s.decision.value_counts()))
    n_rev = n_text = 0
    lens = []
    keys = collections.Counter()
    for js in s.reviews_json:
        try:
            revs = json.loads(js) if js else []
        except Exception:
            continue
        for r in revs:
            n_rev += 1
            keys.update(r.keys())
            text = " ".join(str(r.get(k, "")) for k in r if isinstance(r.get(k), str))
            if len(text.split()) > 50:
                n_text += 1
                lens.append(len(text.split()))
    lens.sort()
    print(f"  reviews: {n_rev}, with >50 words of text: {n_text}, median words: {lens[len(lens)//2] if lens else 0}")
    print("  review fields:", [k for k, _ in keys.most_common(25)])
    print(f"  with full text: {(s.full_text.fillna('').str.len() > 1000).sum()}, with meta-review: {(s.meta_review.fillna('').str.len() > 20).sum()}")
    ex = s[s.reviews_json.notna()].iloc[0]
    print("  example review:", json.dumps(json.loads(ex.reviews_json)[0])[:700])
