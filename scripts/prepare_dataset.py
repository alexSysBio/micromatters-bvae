"""Build the VAE dataset: linescans plus an explicit nucleoid-position channel.

    python scripts/prepare_dataset.py --data-root /path/to/ERC_data --out data/dataset.npz

Channels stored (the training scripts select which to use — the beta-VAE uses 0 and 1):
    0 RplA          mean-normalized linescan
    1 HupA          mean-normalized linescan
    2 position ALL  one Gaussian bump per segmented nucleoid centroid
    3 position MED  a single bump at the median of those centroids

Positions come from `object_scaled_length` (range -1..1 along the cell axis, 0 = midcell)
produced by the 2-D nucleoid segmentation. Both position channels are mean-normalized to 1
exactly like the fluorescence channels, so they are processed identically to a real marker
channel (e.g. the ParB-BFP/parS oriC reporter planned for MICROMATTERS) — a faithful dry run
for adding it.

The two encodings differ only for cells with >1 nucleoid: ALL keeps the segregation geometry
(individual-centroid SD 0.451 in two-nucleoid cells), MED reports where the nucleoid mass is
centred (SD 0.066 in the same cells, because sisters segregate near-symmetrically about
midcell).

Sign convention verified: stored positions correlate +0.91 with the HupA profile centroid in
single-nucleoid cells, i.e. they share the pole orientation of the linescans.

cc_phase, cc_gr and median_position are stored but NEVER used for training — they are held out
to validate what the latent space recovers on its own. The source arrays are the single-cell
data of Papagiannakis et al., eLife 14, RP104276 (2025).
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

SIGMA_BINS = 3.0        # marker width ~ diffraction-limited focus (~100 nm)
N_BINS = 100

FEATURE_COLS = [
    "sym_ribo_0.25", "sym_ribo_0.5", "sym_ribo_0.75",
    "sym_nuc_0.25", "sym_nuc_0.5", "sym_nuc_0.75",
    "newp_ribo_(0, 0.25)", "oldp_ribo_(0, 0.25)",
    "newp_ribo_(0, 0.5)", "oldp_ribo_(0, 0.5)",
    "number_of_objects", "object_asymmetry", "object_distance",
    "mean_fluor_2", "mean_fluor_3", "area_um", "instant_log_gr",
]

BIN_CENTRES = np.linspace(-1.0, 1.0, N_BINS)


def position_channel(positions):
    """Mean-normalized marker profile from nucleoid centre positions (-1..1)."""
    prof = np.zeros(N_BINS, dtype=np.float64)
    sig = SIGMA_BINS * (2.0 / N_BINS)          # sigma in position units
    for p in positions:
        prof += np.exp(-0.5 * ((BIN_CENTRES - float(p)) / sig) ** 2)
    m = prof.mean()
    if not np.isfinite(m) or m <= 0:
        return None
    return prof / m                             # mean == 1, like the other channels


def main(data_root, out_path, val_frac=0.15, seed=0):
    dict_path = f"{data_root}/CJW7323_single_cell_arrays_dict"
    main_path = f"{data_root}/CJW7323_microfluidics_normal_growth"

    print("loading arrays dict...", flush=True)
    with open(dict_path, "rb") as f:
        d = pickle.load(f)

    print("loading main dataframe...", flush=True)
    df = pd.read_pickle(main_path, compression="zip")
    traj_info = df.groupby("cell_trajectory_id").agg(
        cc_gr=("cc_gr", "first"), max_zf=("max_zeroed_frame", "first"))

    # (trajectory, zeroed_frame) -> (positions, hand-crafted features)
    print("indexing per-frame records...", flush=True)
    cols = ["cell_trajectory_id", "zeroed_frame", "object_scaled_length"] + FEATURE_COLS
    rec = {}
    for row in df[cols].itertuples(index=False):
        rec[(row[0], row[1])] = (row[2], np.array(row[3:], dtype=np.float32))

    X, lengths, phases, traj_idx, feats, nobj = [], [], [], [], [], []
    med_pos = []
    traj_ids = []
    skipped = {"profile": 0, "no_record": 0, "no_position": 0}

    for ti, (tid, t) in enumerate(d.items()):
        rib, nuc, cc, ln = t[2][2], t[2][3], t[4], t[5]
        traj_ids.append(tid)
        max_zf = traj_info.loc[tid, "max_zf"]
        for k in range(len(cc)):
            r = np.asarray(rib[k], dtype=np.float32)
            n = np.asarray(nuc[k], dtype=np.float32)
            if (r.shape != (N_BINS,) or n.shape != (N_BINS,) or np.isnan(r).any()
                    or np.isnan(n).any() or not np.isfinite(ln[k])):
                skipped["profile"] += 1
                continue
            key = (tid, int(round(cc[k] * max_zf)))
            if key not in rec:
                skipped["no_record"] += 1
                continue
            pos, fv = rec[key]
            if pos is None or (hasattr(pos, "__len__") and len(pos) == 0):
                skipped["no_position"] += 1
                continue
            pos = [p for p in np.asarray(pos, dtype=float).ravel() if np.isfinite(p)]
            if not pos:
                skipped["no_position"] += 1
                continue
            ch_all = position_channel(pos)
            ch_med = position_channel([float(np.median(pos))])
            if ch_all is None or ch_med is None:
                skipped["no_position"] += 1
                continue
            X.append(np.stack([r, n, ch_all.astype(np.float32),
                               ch_med.astype(np.float32)]))
            med_pos.append(float(np.median(pos)))
            lengths.append(ln[k])
            phases.append(cc[k])
            traj_idx.append(ti)
            feats.append(fv)
            nobj.append(len(pos))

    X = np.asarray(X, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    phases = np.asarray(phases, dtype=np.float32)
    traj_idx = np.asarray(traj_idx, dtype=np.int32)
    feats = np.asarray(feats, dtype=np.float32)
    nobj = np.asarray(nobj, dtype=np.int8)
    med_pos = np.asarray(med_pos, dtype=np.float32)
    cc_gr = traj_info.loc[traj_ids, "cc_gr"].to_numpy(dtype=np.float32)[traj_idx]

    print(f"samples: {len(X):,}  trajectories: {len(traj_ids):,}")
    print("skipped:", skipped)
    print("nucleoids per cell:", {int(k): int(v) for k, v in
                                  zip(*np.unique(nobj, return_counts=True))})
    for i, nm in enumerate(["RplA", "HupA", "position ALL", "position MED"]):
        print(f"  channel {i} ({nm}): mean {X[:, i].mean():.3f}, "
              f"range [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")

    # split by trajectory, so no division cycle appears on both sides
    rng = np.random.default_rng(seed)
    val_traj = set(rng.permutation(len(traj_ids))[: int(val_frac * len(traj_ids))].tolist())
    is_val = np.array([ti in val_traj for ti in traj_idx])
    print(f"train {(~is_val).sum():,} / val {is_val.sum():,} frames")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path, X=X, length=lengths, cc_phase=phases, cc_gr=cc_gr,
        traj_idx=traj_idx, traj_ids=np.array(traj_ids), features=feats,
        feature_names=np.array(FEATURE_COLS), is_val=is_val, n_objects=nobj,
        median_position=med_pos, sigma_bins=SIGMA_BINS,
        channel_names=np.array(["RplA", "HupA", "position_all", "position_median"]))
    print("saved", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True,
                    help="folder holding CJW7323_single_cell_arrays_dict and "
                         "CJW7323_microfluidics_normal_growth")
    ap.add_argument("--out", default="data/dataset.npz")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data_root, a.out, a.val_frac, a.seed)
