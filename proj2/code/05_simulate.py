"""Simulation check: can NUTS recover known decision-model parameters?

Takes the real training papers (their ratings, confidences and years), draws decisions
from the decision model with KNOWN parameters, refits the model on the simulated
decisions, and reports whether each true value falls inside its 95% posterior interval.
Repeats for several parameter settings, including ones where H2 / H3 are true.

Usage (from proj2/):  python code/05_simulate.py
Writes data/processed/decision/simulation.json
"""
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("N_CHAINS", "4")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dec", HERE / "03_decision.py")
dec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dec)

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS, Predictive

OUT = HERE.parent / "data" / "processed" / "decision"
# year thresholds from the real fit, so simulated acceptance rates look like the real ones
TAU = json.load(open(OUT / "result_conf_train2425.json"))["posterior"]["tau"]["mean"]

SETTINGS = {
    # "like the data": vote step at 5 -> 6, small confidence weight, no rebuttal discount
    "data_like": {"v": [-4.5, -3.8, 0.0, 6.5, 11.5, 11.2], "gamma": 0.09, "delta": -0.03, "sigma_a": 0.15},
    # H2 true: initial ratings keep 30% of the weight
    "h2_true": {"v": [-4.5, -3.8, 0.0, 6.5, 11.5, 11.2], "gamma": 0.0, "delta": 0.3, "sigma_a": 0.15},
    # H3 strong and the scale linear (no vote step): confidence 5 counts e^1 = 2.7x confidence 3
    "h3_linear": {"v": [-6.0, -3.0, 0.0, 1.5, 4.5, 7.5], "gamma": 0.5, "delta": 0.0, "sigma_a": 0.3},
}


def to_site(p, d, rng):
    steps = np.diff(np.asarray(p["v"]))
    v0 = p["v"][0]
    # model builds v = [0, cumsum(steps)] - value at index 2, so the steps are what matters
    return {"steps": steps[None].astype(np.float32),
            "gamma": np.array([p["gamma"]], np.float32),
            "delta": np.array([p["delta"]], np.float32),
            "sigma_a": np.array([p["sigma_a"]], np.float32),
            "z_a": rng.standard_normal((1, d["n_area"])).astype(np.float32),
            "tau": np.array([TAU], np.float32)}


def main():
    d, pap = dec.load([2024, 2025], "train")
    rng = np.random.default_rng(1)
    out = {}
    for name, p in SETTINGS.items():
        truth = to_site(p, d, rng)
        sim = Predictive(lambda dd: dec.model(dd, "conf"), posterior_samples=truth, return_sites=["A"])
        A = np.asarray(sim(jax.random.PRNGKey(7), d)["A"][0]).astype(np.int32)
        mcmc = MCMC(NUTS(lambda dd, A: dec.model(dd, "conf", A), target_accept_prob=0.9),
                    num_warmup=500, num_samples=500, num_chains=dec.N_CHAINS,
                    chain_method="parallel", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), d, A=A)
        s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        res = {"accept_rate_simulated": float(A.mean())}
        true_v = np.asarray(p["v"]) - p["v"][2]
        for name_p, true, draws in [("gamma", p["gamma"], s["gamma"]), ("delta", p["delta"], s["delta"]),
                                    ("sigma_a", p["sigma_a"], s["sigma_a"])] + \
                                   [(f"v({k})", true_v[i], s["v"][:, i]) for i, k in enumerate(dec.LEVELS) if k != 5]:
            lo, hi = np.percentile(draws, [2.5, 97.5])
            res[name_p] = {"true": round(float(true), 3), "mean": round(float(draws.mean()), 3),
                           "lo": round(float(lo), 3), "hi": round(float(hi), 3), "covered": bool(lo <= true <= hi)}
        out[name] = res
        print(name, json.dumps(res), flush=True)
    json.dump(out, open(OUT / "simulation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
