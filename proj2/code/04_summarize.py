"""Collect decision-model results into tables: model comparison and the train-year x test-year grid.

Re-scores every saved posterior (data/processed/decision/samples_*.npz) on every test set,
so the grid is complete even when a fit was run with fewer --test years.

Usage (from proj2/):  python code/04_summarize.py
Writes data/processed/decision/summary.json and prints the tables.
"""
import glob
import importlib.util
import json
import os
import re
from pathlib import Path

os.environ.setdefault("N_CHAINS", "4")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dec", HERE / "03_decision.py")
dec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dec)
import numpy as np

OUT = HERE.parent / "data" / "processed" / "decision"
TEXT_COL = {"text": "text_z"}
POINTWISE = {}


def main():
    import sys
    only = sys.argv[1:]  # optional tag prefixes: re-score only these and merge into summary.json
    old = {r["tag"]: r for r in json.load(open(OUT / f"summary{dec.SUFFIX}.json"))} if only and (OUT / f"summary{dec.SUFFIX}.json").exists() else {}
    rows = []
    tests = {}
    for f in sorted(glob.glob(str(OUT / "samples_*.npz"))):
        tag = re.sub(r"^samples_|\.npz$", "", os.path.basename(f))
        if tag in ("smoke", "textsmoke", "text_300", "text_1c", "conf_300", "conf_jit"):
            continue
        is26 = tag.endswith("train26")
        if is26 != (dec.SUFFIX == "_2026"):  # 2026 fits are summarised separately (BML_DATA=_2026)
            continue
        if only and not any(tag.startswith(o) for o in only):
            if tag in old:
                rows.append(old[tag])
            continue
        res_f = OUT / f"result_{tag}.json"
        if not res_f.exists():
            continue
        res = json.load(open(res_f))
        variant = res["model"]
        samples = dict(np.load(f))
        dtr = dec.load(res["train_years"], "train", TEXT_COL.get(variant))[0]
        res["loo"] = dec.psis_loo(samples, dtr, variant)  # recomputed in float64
        POINTWISE[tag] = res["loo"].pop("pointwise")
        row = {"tag": tag, "model": variant, "train": "+".join(str(y) for y in res["train_years"]),
               "n_train": res["n_train"], "max_rhat": round(res["max_rhat"], 3),
               "divergences": res["divergences"], "fit_s": res["fit_seconds"],
               "elpd_loo": round(res["loo"].get("elpd_loo", float("nan")), 1),
               "loo_se": round(res["loo"].get("se", float("nan")), 1)}
        for y in ((2026,) if dec.SUFFIX == "_2026" else (2024, 2025)):
            key = (y, variant)
            if key not in tests:
                tests[key] = dec.load([y], "test", TEXT_COL.get(variant))[0]
            dt = dict(tests[key])
            tr = res["train_years"]
            if y not in tr:
                # unseen year: its threshold was never learned, so use the training year's threshold
                dt["year"] = np.full_like(dt["year"], {2024: 0, 2025: 1, 2026: 0}[tr[0]])
            h = dec.heldout(samples, dt, variant)
            row.update({f"test{y}_elpd": round(h["elpd_per_paper"], 4), f"test{y}_brier": round(h["brier"], 4),
                        f"test{y}_auc": round(h["auc"], 4)})
        post = res["posterior"]
        for p in ("gamma", "delta", "beta", "psi", "b2", "sigma_a"):
            if p in post:
                row[p] = [round(post[p][m], 3) for m in ("mean", "2.5%", "97.5%")]
        if "v" in post:
            row["v"] = {int(k): [round(post["v"][m][i], 2) for m in ("mean", "2.5%", "97.5%")]
                        for i, k in enumerate(dec.LEVELS)}
        if "tau" in post:
            row["tau"] = [[round(post["tau"][m][i], 2) for m in ("mean", "2.5%", "97.5%")] for i in range(2)]
        rows.append(row)
    # paired LOO differences against the conf model fitted on the same years
    for r in rows:
        ref = r["tag"].replace(r["model"] + "_", "conf_", 1)
        if ref in POINTWISE and r["tag"] in POINTWISE and len(POINTWISE[ref]) == len(POINTWISE[r["tag"]]):
            diff = POINTWISE[r["tag"]] - POINTWISE[ref]
            r["loo_diff_vs_conf"] = [round(float(diff.sum()), 1), round(float(np.sqrt(len(diff)) * diff.std()), 1)]
    json.dump(rows, open(OUT / f"summary{dec.SUFFIX}.json", "w"), indent=1)

    cols = ["tag", "n_train", "max_rhat", "divergences", "elpd_loo", "loo_se", "loo_diff_vs_conf"] + \
        [f"test{y}_{m}" for y in ((2026,) if dec.SUFFIX == "_2026" else (2024, 2025)) for m in ("elpd", "auc", "brier")]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in cols))
    print()
    for r in rows:
        print(r["tag"], {k: r[k] for k in ("v", "gamma", "delta", "beta", "psi", "b2", "sigma_a") if k in r})


if __name__ == "__main__":
    main()
