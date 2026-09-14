"""Held-out R2 tables: what does each latent space carry, and under which read-out?

    python -m bvae.evaluate --data data/dataset.npz --models models/

Every `latents_*.npz` found in the models directory is evaluated, so the table grows as you
train more variants. Read-outs:

  Z    latent only                                  -> what the representation itself carries
  B    latent + length
  A    latent + length + log RplA + log HupA        (standard)
  C    latent + length + log ratio

Plus the log ratio recovered from the latent alone, and a linear probe on the latent for each
target — is the information linearly arranged, or does it need a forest to get at?
Writes results_v6.md and eval_v6.npz.
"""
import argparse
import glob
import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

LABELS = {
    "L8": "L8 (length only)",
    "L8c": "L8c (+ conc., learned sigma_s)",
    "L8s01": "L8s01 (+ conc., sigma_s = 0.1)",
    "L8s03": "L8s03 (+ conc., sigma_s = 0.3)",
    "L8s01raw2": "L8s01raw2 (raw 2-ch linescans, sigma_s = 0.1)",
}
ORDER = ["L8", "L8c", "L8s01", "L8s03", "L8s01raw2"]


def load_sorted(path):
    """latent means, columns ordered by decreasing standard deviation."""
    z = np.load(path)["Z"]
    return z[:, np.argsort(-z.std(0))]


def main(data, models_dir, out_md, out_npz):
    d = np.load(data, allow_pickle=True)
    phase, cc_gr, length = d["cc_phase"], d["cc_gr"], d["length"]
    med_pos = d["median_position"].astype(float)
    is_val, traj = d["is_val"], d["traj_idx"]
    X = d["X"]; rib, nuc = X[:, 0], X[:, 1]
    tr, va = ~is_val, is_val
    f = d["features"]; fn = [str(x) for x in d["feature_names"]]
    is1m = np.array(["1minint" in str(t) for t in d["traj_ids"]])[d["traj_idx"]]
    log_r = np.log(f[:, fn.index("mean_fluor_2")] * np.where(is1m, 1.4106, 1.0))
    log_h = np.log(f[:, fn.index("mean_fluor_3")] * np.where(is1m, 1.5862, 1.0))
    ratio = log_r - log_h
    compaction = nuc.var(1)
    poly_asym = rib[:, 50:].mean(1) - rib[:, :50].mean(1)

    RO = {"Z: latent only": None,
          "B: latent + length": np.c_[length],
          "A: latent + length + log RplA + log HupA": np.c_[length, log_r, log_h],
          "C: latent + length + log ratio": np.c_[length, ratio]}

    found = {}
    for p in sorted(glob.glob(os.path.join(models_dir, "latents_*.npz"))):
        tag = os.path.basename(p)[len("latents_"):-len(".npz")]
        found[tag] = p
    tags = [t for t in ORDER if t in found] + [t for t in found if t not in ORDER]
    if not tags:
        raise SystemExit(f"no latents_*.npz in {models_dir}")
    MODELS = {LABELS.get(t, t): load_sorted(found[t]) for t in tags}
    print("evaluating:", ", ".join(tags), flush=True)

    def r2(zm, target, linear=False):
        ok = np.isfinite(target)
        m = (LinearRegression() if linear else
             RandomForestRegressor(100, n_jobs=-1, random_state=0, min_samples_leaf=5))
        m.fit(zm[tr & ok], target[tr & ok])
        return r2_score(target[va & ok], m.predict(zm[va & ok]))

    ntraj = traj.max() + 1
    gr_t = np.zeros(ntraj); val_t = np.zeros(ntraj, bool)
    for i in range(len(traj)):
        gr_t[traj[i]] = cc_gr[i]; val_t[traj[i]] = is_val[i]

    def r2_cycle(zm):
        Zt = np.zeros((ntraj, zm.shape[1])); np.add.at(Zt, traj, zm)
        Zt /= np.bincount(traj, minlength=ntraj)[:, None]
        ok = np.isfinite(gr_t)
        rf = RandomForestRegressor(200, n_jobs=-1, random_state=0)
        rf.fit(Zt[ok & ~val_t], gr_t[ok & ~val_t])
        return r2_score(gr_t[ok & val_t], rf.predict(Zt[ok & val_t]))

    store = {}
    with open(out_md, "w") as out:
        out.write("# Results — concentrations as VAE inputs with fixed scalar noise\n\n")
        out.write("All 8-dim models, same split; held-out R² (RF read-out unless stated).\n\n")
        for ro, sc in RO.items():
            out.write(f"## Read-out {ro}\n\n| model | phase | growth (frame) | "
                      "growth (cycle-avg) | position | compaction | polysome asym. |\n"
                      "|---|---|---|---|---|---|---|\n")
            for nm, z in MODELS.items():
                zm = z if sc is None else np.c_[z, sc]
                row = [r2(zm, phase), r2(zm, cc_gr), r2_cycle(zm), r2(zm, med_pos),
                       r2(zm, compaction), r2(zm, poly_asym)]
                store[(ro, nm)] = row
                print(f"{nm:50s} | {ro:42s} " + "  ".join(f"{v:.3f}" for v in row), flush=True)
                out.write(f"| {nm} | " + " | ".join(f"{v:.3f}" for v in row) + " |\n")
            out.write("\n")

        out.write("## What the latent alone carries: linear probe vs RF, and the log ratio\n\n")
        out.write("| model | log ratio from latent (RF) | phase (linear) | growth (linear) | "
                  "position (linear) | compaction (linear) |\n|---|---|---|---|---|---|\n")
        for nm, z in MODELS.items():
            row = [r2(z, ratio), r2(z, phase, True), r2(z, cc_gr, True),
                   r2(z, med_pos, True), r2(z, compaction, True)]
            store[("probe", nm)] = row
            print(f"{nm:50s} | probes " + "  ".join(f"{v:.3f}" for v in row), flush=True)
            out.write(f"| {nm} | " + " | ".join(f"{v:.3f}" for v in row) + " |\n")

    np.savez_compressed(out_npz, keys=np.array([str(k) for k in store]),
                        vals=np.array([np.array(v) for v in store.values()], dtype=object))
    print(f"wrote {out_md} and {out_npz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/dataset.npz")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out-md", default="results/results_v6.md")
    ap.add_argument("--out-npz", default="results/eval_v6.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out_md) or ".", exist_ok=True)
    main(a.data, a.models, a.out_md, a.out_npz)
