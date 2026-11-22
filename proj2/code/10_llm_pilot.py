"""Pilot: extract review-text features with LLMs on a small sample, and compare models.

Sample: ICLR 2025 training-split reviews whose PRISM text can be matched uniquely to an
ICLR Insights reviewer record (same paper, same final rating, confidence and sub-scores).
Half of the sample raised their rating after the rebuttal, half did not.

The LLM sees only the review text (no rating, confidence, title or decision); explicit
score mentions are masked. It returns:
  body_sentiment   1-10  how positive the main body is (written before the rebuttal)
  update           none / positive / neutral / negative  post-rebuttal addendum, if any
  main_concern     type of the most serious weakness, and whether a rebuttal can fix it
  specificity      1-5   how concrete and technical the review is

Checks against known answers: body_sentiment should track the INITIAL rating more than the
final one; update should track the rating change.

Usage (from proj2/):
  OPENROUTER_API_KEY=... python code/10_llm_pilot.py --models m1,m2 [--n 200] [--repeat 50]
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P1 = "../proj1/data/processed"
RAW = "data/raw"
OUT = "data/processed/llm_pilot"
URL = "https://openrouter.ai/api/v1/chat/completions"

TYPES = ["novelty", "experiments", "theory_correctness", "clarity", "significance_scope", "other"]
SCHEMA = {
    "type": "object",
    "properties": {
        "body_sentiment": {"type": "integer", "minimum": 1, "maximum": 10},
        "update": {"type": "string", "enum": ["none", "positive", "neutral", "negative"]},
        "main_concern_type": {"type": "string", "enum": TYPES},
        "main_concern_fixable": {"type": "boolean"},
        "specificity": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["body_sentiment", "update", "main_concern_type", "main_concern_fixable", "specificity"],
    "additionalProperties": False,
}

SYSTEM = """You read one peer review of a machine-learning conference submission and code it.
Numerical scores have been removed from the text; do not try to guess them.

Return JSON with these fields:
- body_sentiment (1-10): how positive the MAIN BODY of the review is about the paper
  (summary, strengths, weaknesses, questions). Ignore any section added after the author
  response. 1 = clearly recommends rejection with serious flaws; 5 = mixed, leaning negative;
  6 = mixed, leaning positive; 10 = enthusiastic, no real concerns.
- update: did the reviewer add a section after the author response / rebuttal / discussion
  (e.g. "After rebuttal", "Post-rebuttal", "Update", "Thanks for the response")?
  "none" if there is no such section; otherwise whether it moves their view of the paper
  "positive", "neutral" (acknowledges but no change) or "negative".
- main_concern_type: the type of the single most serious weakness the reviewer raises:
  novelty (incremental, already known), experiments (missing baselines, datasets, ablations,
  weak empirical evidence), theory_correctness (errors, unsupported claims, flawed analysis),
  clarity (writing, presentation, missing details), significance_scope (limited impact,
  narrow setting, unrealistic assumptions), other.
- main_concern_fixable: true if that concern could plausibly be resolved during a short
  rebuttal period (new experiment, clarification, added comparison); false if it is about
  the core idea (lack of novelty, fundamental flaw, limited significance).
- specificity (1-5): 1 = generic comments that could apply to many papers; 5 = detailed,
  technical, points to specific equations, experiments, numbers or related papers."""

SCORE_RE = re.compile(
    r"(?i)(?P<pre>\b(?:score|rating|rate|scored|raise[sd]?|increas(?:e|ed|ing)|lower(?:ed)?|decreas(?:e|ed)"
    r"|upgrade[sd]?|adjust(?:ed)?|change[sd]?|maintain(?:ed)?|keep|kept)\b[^.\n]{0,40}?)"
    r"\b(?P<num>\d{1,2}(?:\.\d)?)\b(?:\s*(?:/|out of)\s*10)?")
LEAD_RE = re.compile(r"(?im)^\s*(rating|score|confidence)\s*:.*$")
ARROW_RE = re.compile(r"(?i)\b\d{1,2}\s*(?:->|→|to)\s*\d{1,2}\b")


def mask(text):
    text = LEAD_RE.sub("", text)
    text = ARROW_RE.sub("[X] to [X]", text)
    for _ in range(3):  # a phrase can hold several numbers
        text = SCORE_RE.sub(lambda m: m.group("pre") + "[X]", text)
    return text


def review_text(r):
    parts = []
    for k in ("Summary", "Strengths", "Weaknesses", "Questions", "Limitations"):
        v = str(r.get(k) or "").strip()
        if v:
            parts.append(f"## {k}\n{v}")
    return mask("\n\n".join(parts))


def build_sample(n, seed=0):
    rev = pd.read_parquet(f"{P1}/reviews.parquet")
    rev = rev[(rev.split == "train") & rev.in_analysis & rev.r0.notna() & rev.r1.notna()]
    pr = pd.read_parquet(f"{RAW}/prism_papers.parquet", columns=["paper_id", "venue", "reviews_json"])
    pr = pr[(pr.venue == "ICLR_2025") & pr.paper_id.isin(set(rev.paper_id))]
    key = ["r1", "c1", "snd1", "prs1", "ctb1"]
    rows = []
    by_paper = {pid: g for pid, g in rev.groupby("paper_id")}
    for pid, js in zip(pr.paper_id, pr.reviews_json):
        ins = by_paper[pid]
        ins_keys = ins[key].astype(float).apply(tuple, axis=1)
        for r in json.loads(js):
            try:
                k = tuple(float(r[f]) for f in ("Rating", "Confidence", "Soundness", "Presentation", "Contribution"))
            except (KeyError, ValueError, TypeError):
                continue
            hit = ins[ins_keys == k]
            if len(hit) == 1:
                rows.append({"paper_id": pid, "review_id": r.get("Review ID"),
                             **hit.iloc[0][["reviewer_id", "r0", "r1", "c0", "c1"]].to_dict(),
                             "text": review_text(r)})
    df = pd.DataFrame(rows)
    df = df[~df.duplicated(["paper_id", "reviewer_id"], keep=False)]
    df["dr"] = df.r1 - df.r0
    print(f"matchable training reviews: {len(df)} (of {len(rev)})")
    raised = df[df.dr > 0].sample(n // 2, random_state=seed)
    same = df[df.dr == 0].sample(n - n // 2, random_state=seed)
    return pd.concat([raised, same]).sample(frac=1, random_state=seed).reset_index(drop=True)


MANDATORY_REASONING = set()  # models whose endpoint refuses reasoning off


def call(model, text, key, timeout=180):
    r = call_once(model, text, key, timeout, reasoning=model not in MANDATORY_REASONING)
    if "Reasoning is mandatory" in str(r.get("error", "")):
        MANDATORY_REASONING.add(model)
        r = call_once(model, text, key, timeout, reasoning=True)
    return r


def call_once(model, text, key, timeout, reasoning):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 4000 if reasoning else 400,
        "reasoning": {"effort": "low"} if reasoning else {"enabled": False},
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "review_codes", "strict": True, "schema": SCHEMA}},
        "usage": {"include": True},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2 ** attempt * 3)
                continue
            return {"error": f"HTTP {e.code}: {msg}", "latency": time.time() - t0}
        except Exception as e:  # timeouts, connection resets
            if attempt < 3:
                time.sleep(2 ** attempt * 3)
                continue
            return {"error": repr(e), "latency": time.time() - t0}
    lat = time.time() - t0
    try:
        content = out["choices"][0]["message"]["content"]
        content = re.sub(r"^```(json)?|```$", "", content.strip()).strip()
        codes = json.loads(content)
    except Exception as e:
        return {"error": f"parse: {e!r}: {str(out)[:300]}", "latency": lat}
    u = out.get("usage", {})
    return {**codes, "latency": lat, "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "cost": u.get("cost"),
            "provider": out.get("provider")}


def run(model, df, key, workers):
    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda t: call(model, t, key), df.text))
    wall = time.time() - t0
    out = pd.concat([df.drop(columns="text").reset_index(drop=True), pd.DataFrame(res)], axis=1)
    return out, wall


def summarize(model, out, wall):
    ok = out[out["error"].isna()] if "error" in out else out
    if not len(ok):
        return {"model": model, "n": len(out), "ok": 0,
                "errors": out["error"].str[:200].value_counts().head(3).to_dict()}
    upd = ok["update"].map({"none": 0, "neutral": 0, "positive": 1, "negative": -1})
    s = {
        "model": model, "n": len(out), "ok": len(ok),
        "wall_s": round(wall, 1), "median_latency_s": round(out.latency.median(), 2),
        "p90_latency_s": round(out.latency.quantile(0.9), 2),
        "cost_usd": round(float(ok.cost.fillna(0).sum()), 4) if "cost" in ok else None,
        "mean_prompt_tokens": round(float(ok.prompt_tokens.mean()), 0),
        "mean_completion_tokens": round(float(ok.completion_tokens.mean()), 0),
        "rho_sent_r0": round(spearmanr(ok.body_sentiment, ok.r0)[0], 3),
        "rho_sent_r1": round(spearmanr(ok.body_sentiment, ok.r1)[0], 3),
        "rho_update_dr": round(spearmanr(upd, ok.dr)[0], 3),
        "update_any_when_raised": round(float((ok[ok.dr > 0]["update"] != "none").mean()), 3),
        "update_any_when_same": round(float((ok[ok.dr == 0]["update"] != "none").mean()), 3),
        "update_pos_when_raised": round(float((ok[ok.dr > 0]["update"] == "positive").mean()), 3),
        "update_pos_when_same": round(float((ok[ok.dr == 0]["update"] == "positive").mean()), 3),
        "concern_types": ok.main_concern_type.value_counts().to_dict(),
        "fixable_share": round(float(ok.main_concern_fixable.mean()), 3),
        "rho_specificity_c0": round(spearmanr(ok.specificity, ok.c0)[0], 3),
        "providers": ok.provider.value_counts().to_dict() if "provider" in ok else None,
        "errors": out.error.dropna().str[:120].value_counts().head(3).to_dict() if "error" in out else {},
    }
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"].strip()
    os.makedirs(OUT, exist_ok=True)
    df = build_sample(a.n)
    df.to_parquet(f"{OUT}/sample.parquet")
    print(f"sample: {len(df)} reviews, median {int(df.text.str.split().str.len().median())} words")
    summaries, outs = [], {}
    for m in a.models.split(","):
        out, wall = run(m, df, key, a.workers)
        tag = re.sub(r"[^A-Za-z0-9]+", "_", m)
        out.to_parquet(f"{OUT}/codes_{tag}.parquet")
        s = summarize(m, out, wall)
        if a.repeat:  # test-retest on the first `repeat` reviews
            rep, _ = run(m, df.head(a.repeat), key, a.workers)
            both = out.head(a.repeat).join(rep[["body_sentiment", "update", "main_concern_type", "specificity"]], rsuffix="_rep")
            both = both.dropna(subset=["body_sentiment", "body_sentiment_rep"])
            s["retest_sent_exact"] = round(float((both.body_sentiment == both.body_sentiment_rep).mean()), 3)
            s["retest_sent_within1"] = round(float(((both.body_sentiment - both.body_sentiment_rep).abs() <= 1).mean()), 3)
            s["retest_update_agree"] = round(float((both["update"] == both.update_rep).mean()), 3)
            s["retest_concern_agree"] = round(float((both.main_concern_type == both.main_concern_type_rep).mean()), 3)
        summaries.append(s)
        outs[m] = out
        print(json.dumps(s, indent=1, default=str), flush=True)
    if len(outs) == 2:
        (m1, o1), (m2, o2) = outs.items()
        j = o1.join(o2[["body_sentiment", "update", "main_concern_type", "main_concern_fixable", "specificity"]], rsuffix="_b").dropna(subset=["body_sentiment", "body_sentiment_b"])
        cross = {
            "sentiment_rho": round(spearmanr(j.body_sentiment, j.body_sentiment_b)[0], 3),
            "update_agree": round(float((j["update"] == j.update_b).mean()), 3),
            "concern_agree": round(float((j.main_concern_type == j.main_concern_type_b).mean()), 3),
            "fixable_agree": round(float((j.main_concern_fixable == j.main_concern_fixable_b).mean()), 3),
            "specificity_rho": round(spearmanr(j.specificity, j.specificity_b)[0], 3),
        }
        print("cross-model agreement:", cross)
        summaries.append({"cross_model": cross})
    json.dump(summaries, open(f"{OUT}/summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
