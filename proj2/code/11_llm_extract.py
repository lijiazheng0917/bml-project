"""Code every analysis review with an LLM (full run of the pilot in 10_llm_pilot.py).

Same prompt, schema, masking and model call as the pilot. Results are appended to a JSONL
checkpoint, so the script can be stopped and restarted; finished reviews are skipped.

Usage (from proj2/):
  OPENROUTER_API_KEY=... python code/11_llm_extract.py --model deepseek/deepseek-v4.1-flash-1017:nitro --workers 64
Output: data/processed/llm_codes.parquet (one row per year, paper_id, reviewer_id)
"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import threading
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pilot", HERE / "10_llm_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

PROC = HERE.parent / "data" / "processed"
CKPT = PROC / "llm_codes.jsonl"
OUT = PROC / "llm_codes.parquet"
REVIEWS = PROC / "reviews.parquet"
KEYS = ["year", "paper_id", "reviewer_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suffix", default="", help="e.g. _2026: use reviews_2026.parquet and llm_codes_2026.*")
    a = ap.parse_args()
    global CKPT, OUT, REVIEWS
    if a.suffix:
        CKPT, OUT = PROC / f"llm_codes{a.suffix}.jsonl", PROC / f"llm_codes{a.suffix}.parquet"
        REVIEWS = PROC / f"reviews{a.suffix}.parquet"
    key = os.environ["OPENROUTER_API_KEY"].strip()

    rev = pd.read_parquet(REVIEWS, columns=KEYS + ["text", "in_analysis"])
    rev = rev[rev.in_analysis.fillna(False) & rev.text.notna()].drop(columns="in_analysis")
    done = set()
    if CKPT.exists():
        for line in open(CKPT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line cut off when a run was stopped
                continue
            if "error" not in r:
                done.add((r["year"], r["paper_id"], r["reviewer_id"]))
    todo = rev[[(y, p, r) not in done for y, p, r in zip(rev.year, rev.paper_id, rev.reviewer_id)]]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"reviews: {len(rev)}, done: {len(done)}, to do: {len(todo)}", flush=True)

    lock = threading.Lock()
    stats = {"n": 0, "err": 0, "cost": 0.0}
    t0 = time.time()
    f = open(CKPT, "a")

    def work(row):
        res = pilot.call(a.model, pilot.mask(row.text), key)
        rec = {"year": int(row.year), "paper_id": row.paper_id, "reviewer_id": row.reviewer_id,
               "model": a.model, **res}
        with lock:
            f.write(json.dumps(rec) + "\n")
            stats["n"] += 1
            stats["err"] += "error" in res
            stats["cost"] += res.get("cost") or 0.0
            if stats["n"] % 2000 == 0:
                f.flush()
                el = time.time() - t0
                print(f"{stats['n']}/{len(todo)}  errors {stats['err']}  ${stats['cost']:.2f}  "
                      f"{stats['n'] / el:.1f}/s  eta {(len(todo) - stats['n']) / (stats['n'] / el) / 60:.1f} min", flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo.itertuples(index=False)))
    f.close()
    print(f"finished {stats['n']} in {(time.time() - t0) / 60:.1f} min, errors {stats['err']}, cost ${stats['cost']:.2f}", flush=True)

    # collect the latest successful code per review
    recs = []
    for l in open(CKPT):
        try:
            recs.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    df = pd.DataFrame([r for r in recs if "error" not in r]).drop_duplicates(KEYS, keep="last")
    df.to_parquet(OUT, index=False)
    print(f"saved {len(df)} coded reviews to {OUT} (of {len(rev)})", flush=True)


if __name__ == "__main__":
    main()
