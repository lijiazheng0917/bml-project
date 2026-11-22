"""Bayesian decision models for the AC: how ratings are combined into accept / reject.

Each paper i has J_i reviewers with final rating r1, initial rating r0 (both in
{1,3,5,6,8,10}), final confidence c1, and (optionally) an LLM text-sentiment code.

  score    S_i = sum_j w_ij [ (1-delta) v(r1_ij) + delta v(r0_ij) ] + beta * text_i
  weights  w_ij = softmax_j(gamma * c1_ij)           (equal weights when gamma = 0)
  accept   A_i ~ Bernoulli( logistic( S_i - tau_{y_i} - alpha_{a_i} ) )
           alpha_a ~ N(0, sigma_a^2) area offsets, tau_y year thresholds

Variants (nested):
  mean   v(k) = b * k (linear), gamma = delta = beta = 0      -- "use the mean rating"
  votes  linear mean + b2 * (share of reviewers rating >= 6)
  free   free v(k) with v(5) = 0
  rebut  free + delta
  conf   free + delta + gamma
  text   free + delta + gamma + beta   (needs text codes)
  cons   free + delta + gamma + psi * sd(final ratings)   (votes versus consensus)

Fits with NUTS on the training papers of the chosen years; reports posterior summaries,
PSIS-LOO on the training data, and held-out metrics on the test papers.

Usage (from proj2/):  python code/03_decision.py --model conf --train 2024,2025 --test 2024,2025
"""
import argparse
import json
import os
import time

N_CHAINS = int(os.environ.get("N_CHAINS", "4"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={N_CHAINS}")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import log_likelihood
from scipy.stats import rankdata

numpyro.enable_x64(False)
# BML_DATA="" -> ICLR 2024+2025 tables; BML_DATA="_2026" -> ICLR 2026 tables (other rating scale).
# In both scales index 2 is the reference value (5 in 2024/25, 4 in 2026), and index 3 is 6.
SUFFIX = os.environ.get("BML_DATA", "")
LEVELS = np.array([0, 2, 4, 6, 8, 10]) if SUFFIX == "_2026" else np.array([1, 3, 5, 6, 8, 10])
LEVEL_IDX = {v: i for i, v in enumerate(LEVELS)}
JMAX = 8
PROC = "data/processed"
OUT = "data/processed/decision"


# ---------------------------------------------------------------- data
def load(years, split, text_col=None):
    pap = pd.read_parquet(f"{PROC}/papers{SUFFIX}.parquet")
    rev = pd.read_parquet(f"{PROC}/reviews{SUFFIX}.parquet")
    if text_col:
        # LLM body sentiment, standardised within (year, initial rating): the text beyond the score.
        codes = pd.read_parquet(f"{PROC}/llm_codes{SUFFIX}.parquet", columns=["year", "paper_id", "reviewer_id", "body_sentiment"])
        rev = rev.merge(codes, on=["year", "paper_id", "reviewer_id"], how="left")
        g = rev.groupby(["year", "r0"]).body_sentiment
        rev["text_z"] = ((rev.body_sentiment - g.transform("mean")) / g.transform("std")).fillna(0.0)
    pap = pap[pap.year.isin(years) & (pap.split == split) & pap.in_analysis].reset_index(drop=True)
    rev = rev.merge(pap[["year", "paper_id"]].reset_index().rename(columns={"index": "row"}),
                    on=["year", "paper_id"])
    rev = rev.sort_values(["row", "reviewer_id"])
    rev["k"] = rev.groupby("row").cumcount()
    rev = rev[rev.k < JMAX]
    n = len(pap)
    shape = (n, JMAX)
    mask = np.zeros(shape, bool)
    r1 = np.full(shape, 2, np.int32)   # pad with index of rating 5
    r0 = np.full(shape, 2, np.int32)
    c1 = np.zeros(shape, np.float32)
    tx = np.zeros(shape, np.float32)
    rr, kk = rev.row.to_numpy(), rev.k.to_numpy()
    mask[rr, kk] = True
    r1[rr, kk] = rev.r1.map(LEVEL_IDX).to_numpy()
    r0[rr, kk] = rev.r0.map(LEVEL_IDX).to_numpy()
    c1[rr, kk] = rev.c1.fillna(3).to_numpy() - 3.0   # centred at 3
    if text_col:
        tx[rr, kk] = rev[text_col].fillna(0).to_numpy()
    areas = sorted(pd.read_parquet(f"{PROC}/papers{SUFFIX}.parquet").primary_area.dropna().unique())
    a_idx = pap.primary_area.map({a: i for i, a in enumerate(areas)}).fillna(len(areas)).astype(int).to_numpy()
    y_idx = pap.year.map({2024: 0, 2025: 1, 2026: 0}).to_numpy()
    return dict(mask=mask, r1=r1, r0=r0, c1=c1, tx=tx, area=a_idx, year=y_idx,
                n_area=len(areas) + 1, A=pap.accepted.to_numpy().astype(np.int32)), pap


# ---------------------------------------------------------------- model
N_AREA = int(pd.read_parquet(f"{PROC}/papers{SUFFIX}.parquet", columns=["primary_area"]).primary_area.dropna().nunique()) + 1


def model(d, variant, A=None):
    mask = jnp.asarray(d["mask"])
    mf = mask.astype(jnp.float32)
    n_rev = mf.sum(1)
    lv = jnp.asarray(LEVELS, jnp.float32)

    tau = numpyro.sample("tau", dist.Normal(0, 3).expand([2]))
    sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(1.0))
    with numpyro.plate("areas", N_AREA):  # a Python constant, so the data can be traced
        z_a = numpyro.sample("z_a", dist.Normal(0, 1))
    alpha = sigma_a * z_a

    if variant in ("mean", "votes"):
        b = numpyro.sample("b", dist.Normal(0, 3))
        v = b * (lv - lv[2])
    else:
        # free values, v(5) = 0; written as steps so the order is not forced
        steps = numpyro.sample("steps", dist.Normal(0, 3).expand([5]))
        cs = jnp.cumsum(steps)
        v = jnp.concatenate([jnp.zeros(1), cs])
        v = v - v[2]
    numpyro.deterministic("v", v)

    gamma = numpyro.sample("gamma", dist.Normal(0, 1)) if variant in ("conf", "text", "cons") else 0.0
    delta = numpyro.sample("delta", dist.Normal(0, 0.5)) if variant in ("rebut", "conf", "text", "cons") else 0.0

    logits_w = gamma * jnp.asarray(d["c1"]) + jnp.where(mask, 0.0, -1e9)
    w = jax.nn.softmax(logits_w, axis=1)
    vr1 = v[jnp.asarray(d["r1"])]
    vr0 = v[jnp.asarray(d["r0"])]
    S = jnp.sum(w * ((1 - delta) * vr1 + delta * vr0), axis=1)

    if variant == "votes":
        b2 = numpyro.sample("b2", dist.Normal(0, 3))
        share6 = jnp.sum(mf * (jnp.asarray(d["r1"]) >= 3), 1) / n_rev
        S = S + b2 * share6
    if variant == "cons":
        # disagreement penalty: standard deviation of the final ratings (H1: votes versus consensus)
        psi = numpyro.sample("psi", dist.Normal(0, 1))
        rv = lv[jnp.asarray(d["r1"])]
        m = jnp.sum(mf * rv, 1) / n_rev
        sd = jnp.sqrt(jnp.sum(mf * (rv - m[:, None]) ** 2, 1) / n_rev)
        S = S - psi * sd
    if variant == "text":
        beta = numpyro.sample("beta", dist.Normal(0, 1))
        S = S + beta * jnp.sum(w * jnp.asarray(d["tx"]), axis=1)

    eta = S - tau[jnp.asarray(d["year"])] - alpha[jnp.asarray(d["area"])]
    with numpyro.plate("papers", mask.shape[0]):
        numpyro.sample("A", dist.Bernoulli(logits=eta), obs=A)


# ---------------------------------------------------------------- metrics
def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def chunked_ll(samples, d, variant, A, chunk=250):
    """Log-likelihood of A for every posterior draw, computed `chunk` draws at a time.

    Evaluating all 4,000 draws at once builds (draws x papers x reviewers) arrays of several GB
    each, which ran out of memory for the text model."""
    n = len(next(iter(samples.values())))
    out = []
    for s0 in range(0, n, chunk):
        part = {k: v[s0:s0 + chunk] for k, v in samples.items()}
        ll = log_likelihood(lambda dd, A: model(dd, variant, A), part, d, A=A)["A"]
        out.append(np.asarray(ll, dtype=np.float64))
    return np.concatenate(out, 0)


def predict_p(samples, d, variant):
    # use the log-likelihood of A=1 to get exact probabilities instead of Bernoulli draws
    ll1 = chunked_ll(samples, d, variant, np.ones(len(d["A"]), np.int32))
    return np.exp(np.asarray(ll1))          # (draws, papers)


def heldout(samples, d, variant):
    P = predict_p(samples, d, variant)
    y = d["A"]
    lik = np.where(y == 1, P, 1 - P)        # (draws, papers)
    lpd = np.log(lik.mean(0) + 1e-12)       # log pointwise predictive density
    p = P.mean(0)
    return {"n": int(len(y)), "elpd_per_paper": float(lpd.mean()),
            "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p),
            "calibration": calib(y, p)}


def calib(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.sum():
            out.append([round(float(p[m].mean()), 4), round(float(y[m].mean()), 4), int(m.sum())])
    return out


def psis_loo(samples, d, variant):
    import arviz as az
    ll = chunked_ll(samples, d, variant, d["A"])
    ll = np.asarray(ll, dtype=np.float64).reshape(N_CHAINS, -1, ll.shape[-1])
    idata = az.from_dict(posterior={"_": np.zeros(ll.shape[:2])}, log_likelihood={"A": ll})
    res = az.loo(idata, pointwise=True)
    k = np.asarray(res.pareto_k)
    return {"elpd_loo": float(res.elpd_loo), "se": float(res.se), "p_loo": float(res.p_loo),
            "n_bad_k": int((k > 0.7).sum()), "pointwise": np.asarray(res.loo_i)}


# ---------------------------------------------------------------- main
T_START = time.time()


def log(msg):
    """Timestamped progress line, flushed at once so a stuck stage is visible in the log."""
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T_START:6.1f}s] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", default="2024,2025")
    ap.add_argument("--test", default="2024,2025")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tr_years = [int(x) for x in a.train.split(",")]
    te_years = [int(x) for x in a.test.split(",")] if a.test else []
    d, pap = load(tr_years, "train", a.text_col)
    tag = a.tag or f"{a.model}_train{''.join(str(y)[2:] for y in tr_years)}"
    print(f"[{tag}] training papers: {len(pap)}", flush=True)

    t0 = time.time()
    # start every chain near the prior median (a random start of gamma = +-2 can trap a chain)
    mcmc = MCMC(NUTS(lambda dd, A: model(dd, a.model, A), target_accept_prob=0.9,
                     init_strategy=init_to_median(num_samples=15), max_tree_depth=8),
                num_warmup=a.warmup, num_samples=a.samples, num_chains=N_CHAINS,
                chain_method=os.environ.get("CHAIN_METHOD", "sequential"), progress_bar=False,
                # pass the data as traced inputs: closing over the arrays made XLA fold them into
                # constants, which took minutes and tens of GB to compile
                jit_model_args=True)
    log(f"{tag}: NUTS start ({N_CHAINS} chains, {a.warmup}+{a.samples})")
    mcmc.run(jax.random.PRNGKey(0), d, A=d["A"])
    jax.block_until_ready(mcmc.get_samples())  # JAX runs asynchronously; wait for the real end
    fit_s = time.time() - t0
    log(f"{tag}: NUTS done in {fit_s:.1f}s")
    samples = mcmc.get_samples()
    grouped = mcmc.get_samples(group_by_chain=True)
    log(f"{tag}: samples collected")

    from numpyro.diagnostics import summary
    summ = summary({k: np.asarray(v) for k, v in grouped.items() if k not in ("z_a",)}, prob=0.95)
    post = {}
    for k, s in summ.items():
        post[k] = {m: np.asarray(s[m]).round(4).tolist() for m in ("mean", "std", "2.5%", "97.5%", "n_eff", "r_hat")}
    n_div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    log(f"{tag}: summary done, divergences {n_div}")

    result = {"tag": tag, "model": a.model, "train_years": tr_years, "n_train": len(pap),
              "fit_seconds": round(fit_s, 1), "divergences": n_div,
              "max_rhat": float(max(np.nanmax(np.asarray(s["r_hat"])) for s in summ.values())),
              "posterior": post}
    try:
        log(f"{tag}: PSIS-LOO start")
        result["loo"] = {k: v for k, v in psis_loo(samples, d, a.model).items() if k != "pointwise"}
        log(f"{tag}: PSIS-LOO done")
    except Exception as e:  # keep going if arviz changes its API
        result["loo"] = {"error": repr(e)}
    result["train_fit"] = {k: v for k, v in heldout(samples, d, a.model).items() if k != "calibration"}
    log(f"{tag}: training-set scores done")
    for y in te_years:
        log(f"{tag}: scoring test {y}")
        dt, pt = load([y], "test", a.text_col)
        result[f"test_{y}"] = heldout(samples, dt, a.model)
    np.savez_compressed(f"{OUT}/samples_{tag}.npz", **{k: np.asarray(v) for k, v in samples.items()})
    json.dump(result, open(f"{OUT}/result_{tag}.json", "w"), indent=1)
    brief = {k: result[k] for k in ("tag", "n_train", "fit_seconds", "divergences", "max_rhat", "loo")}
    brief.update({k: {m: result[k][m] for m in ("elpd_per_paper", "brier", "auc")} for k in result if k.startswith("test_")})
    brief.update({p: {m: post[p][m] for m in ("mean", "2.5%", "97.5%")} for p in ("v", "gamma", "delta", "b", "b2", "beta", "sigma_a") if p in post})
    print(json.dumps(brief, indent=1), flush=True)


if __name__ == "__main__":
    main()
