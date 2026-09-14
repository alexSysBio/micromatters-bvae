"""Held-out R2 for a named scalar feature set, optionally + length and + latent.

    python -m bvae.features '["logR","logH"]'
    python -m bvae.features '["latent","length"]' --seed 0 --leaf 5 --jobs 2

Feature names: logR, logH, ratio, sum, length, latent (the 8 sorted latent means).
Prints one JSON line: {"features": [...], "results": {target: {"rf": x, "linear": x}}}
Targets: gr_frame (growth rate per frame), gr_cycle (cycle-averaged), phase.
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def main(names, data, latents, seed, leaf, jobs):
    d = np.load(data, allow_pickle=True)
    gr, phase, length = d["cc_gr"], d["cc_phase"], d["length"]
    is_val, traj = d["is_val"], d["traj_idx"]
    f = d["features"]; fn = [str(x) for x in d["feature_names"]]
    is1m = np.array(["1minint" in str(t) for t in d["traj_ids"]])[traj]
    logR = np.log(f[:, fn.index("mean_fluor_2")] * np.where(is1m, 1.4106, 1.0))
    logH = np.log(f[:, fn.index("mean_fluor_3")] * np.where(is1m, 1.5862, 1.0))
    n = np.load(latents); Z = n["Z"][:, np.argsort(-n["Z"].std(0))]
    COLS = {"logR": logR[:, None], "logH": logH[:, None],
            "ratio": (logR - logH)[:, None], "sum": (logR + logH)[:, None],
            "length": length[:, None], "latent": Z}
    X = np.hstack([COLS[nm] for nm in names])
    tr, va = ~is_val, is_val
    ntraj = traj.max() + 1
    gr_t = np.zeros(ntraj); val_t = np.zeros(ntraj, bool)
    gr_t[traj] = gr; val_t[traj] = is_val

    def fit(m, Xf, y):
        ok = np.isfinite(y)
        m.fit(Xf[tr & ok], y[tr & ok])
        return r2_score(y[va & ok], m.predict(Xf[va & ok]))

    out = {}
    for tgt, y in (("gr_frame", gr), ("phase", phase)):
        rf = RandomForestRegressor(100, n_jobs=jobs, random_state=seed, min_samples_leaf=leaf)
        out[tgt] = {"rf": round(fit(rf, X, y), 4),
                    "linear": round(fit(LinearRegression(), X, y), 4)}
    Xt = np.zeros((ntraj, X.shape[1])); np.add.at(Xt, traj, X)
    Xt /= np.bincount(traj, minlength=ntraj)[:, None]
    ok = np.isfinite(gr_t)
    rf = RandomForestRegressor(200, n_jobs=jobs, random_state=seed)
    rf.fit(Xt[ok & ~val_t], gr_t[ok & ~val_t])
    lin = LinearRegression().fit(Xt[ok & ~val_t], gr_t[ok & ~val_t])
    out["gr_cycle"] = {
        "rf": round(r2_score(gr_t[ok & val_t], rf.predict(Xt[ok & val_t])), 4),
        "linear": round(r2_score(gr_t[ok & val_t], lin.predict(Xt[ok & val_t])), 4)}
    print(json.dumps({"features": names, "seed": seed, "leaf": leaf, "results": out}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("features", help='JSON list, e.g. \'["latent","length"]\'')
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--latents", default="models/latents_L8s01raw2.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leaf", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    main(json.loads(a.features), a.data, a.latents, a.seed, a.leaf, a.jobs)
