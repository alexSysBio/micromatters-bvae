"""Train the beta-VAE with the scalar observation noise fixed rather than learned.

    python -m bvae.train 0.1 --raw2          # L8s01raw2: raw 2-channel linescans
    python -m bvae.train 0.1 0.3             # sym/anti channels, one model per sigma

Feeding the RplA and HupA concentrations to the encoder does not by itself put them in the
latent space. With a LEARNED observation noise the ELBO's cheapest solution is to inflate
sigma_s on three numbers and spend the capacity on the 400 profile values, and the latent
discards the concentrations. Fixing sigma_s (in standardised units) instead of learning it is
what changes the representation.

Everything else is held constant across models: scalars [length, log mean RplA, log mean HupA]
standardised on the training split, 8 latent dims, beta = 2, 40 epochs, batch 512, Adam at
1e-3, seed 0, and the same trajectory-level train/validation split.
Tags: L8s01 (sigma_s = 0.1), L8s03 (0.3), with `raw2` appended for the 2-channel variant.
"""
import argparse
import os

import numpy as np
import torch

from bvae.model import VAE, elbo_terms, sym_anti

BETA, EPOCHS, BATCH, LR, SEED, ZDIM = 2.0, 40, 512, 1e-3, 0, 8


def scalars_v6(d):
    """[length, log corrected mean RplA, log corrected mean HupA] in measured units.

    The 1.4106 / 1.5862 factors correct the trajectories imaged at a 1-minute interval onto
    the same fluorescence scale as the rest.
    """
    f = d["features"]
    fn = [str(x) for x in d["feature_names"]]
    is1m = np.array(["1minint" in str(t) for t in d["traj_ids"]])[d["traj_idx"]]
    log_r = np.log(f[:, fn.index("mean_fluor_2")] * np.where(is1m, 1.4106, 1.0))
    log_h = np.log(f[:, fn.index("mean_fluor_3")] * np.where(is1m, 1.5862, 1.0))
    return np.c_[d["length"], log_r, log_h]


def main(sig_s, data, out_dir, raw2=False):
    tag = f"L8s{sig_s:.1f}".replace("0.", "0") + ("raw2" if raw2 else "")   # 0.1 -> L8s01
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else ("cuda" if torch.cuda.is_available() else "cpu"))
    d = np.load(data, allow_pickle=True)
    X4 = ((d["X"][:, [0, 1]] - 1.0).astype(np.float32) if raw2
          else sym_anti(d["X"][:, [0, 1]] - 1.0))
    n_ch = X4.shape[1]
    is_val = d["is_val"]
    raw = scalars_v6(d)
    m_, s_ = raw[~is_val].mean(0), raw[~is_val].std(0)
    S = ((raw - m_) / s_).astype(np.float32)

    Xtr, Ltr = torch.from_numpy(X4[~is_val]), torch.from_numpy(S[~is_val])
    Xva, Lva = torch.from_numpy(X4[is_val]).to(dev), torch.from_numpy(S[is_val]).to(dev)
    model = VAE(ZDIM, channels=n_ch, n_scalars=3).to(dev)
    # fixed, known scalar observation noise (not learned)
    model.log_sig2_l.data.fill_(float(np.log(sig_s ** 2)))
    model.log_sig2_l.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    n = len(Xtr)
    print(f"[{tag}] zdim={ZDIM} channels={n_ch} scalars=[length, log RplA, log HupA] "
          f"sigma_s FIXED {sig_s} train={n:,} val={len(Xva):,} device={dev}", flush=True)
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i: i + BATCH]
            x, l = Xtr[idx].to(dev), Ltr[idx].to(dev)
            xhat, lhat, mu, logvar = model(x, l)
            nll_x, nll_l, kl = elbo_terms(model, x, l, xhat, lhat, mu, logvar)
            loss = nll_x + nll_l + BETA * kl
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                xhat, lhat, mu, logvar = model(Xva, Lva)
                nll_x, nll_l, kl = elbo_terms(model, Xva, Lva, xhat, lhat, mu, logvar)
                rmse = ((lhat - Lva) ** 2).mean(0).sqrt()
            print(f"[{tag}] ep {ep:3d} val_elbo {(nll_x + nll_l + BETA * kl).item():9.2f} "
                  f"KL {kl.item():6.2f} scalar RMSE (SD units) " +
                  " ".join(f"{v:.3f}" for v in rmse.tolist()), flush=True)

    model.eval()
    mus, sds = [], []
    with torch.no_grad():
        Xall, Lall = torch.from_numpy(X4), torch.from_numpy(S)
        for i in range(0, len(Xall), 4096):
            mu, logvar = model.encode(Xall[i: i + 4096].to(dev), Lall[i: i + 4096].to(dev))
            mus.append(mu.cpu().numpy()); sds.append(np.exp(0.5 * logvar.cpu().numpy()))
    Z, SIG = np.concatenate(mus), np.concatenate(sds)
    print(f"[{tag}] latent SD:", np.round(Z.std(0), 3))
    torch.save({"state_dict": model.state_dict(), "len_mu": float(m_[0]), "len_sd": float(s_[0]),
                "scalar_mu": m_, "scalar_sd": s_, "scalar_names": ["length", "log_rpla", "log_hupa"],
                "sigma_s_fixed": sig_s, "latent_dim": ZDIM, "beta": BETA, "channels": n_ch,
                "n_scalars": 3}, f"{out_dir}/vae_{tag}.pt")
    np.savez_compressed(f"{out_dir}/latents_{tag}.npz", Z=Z, SIG=SIG)
    print(f"[{tag}] saved vae_{tag}.pt and latents_{tag}.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sigma", nargs="*", type=float, default=[0.1, 0.3],
                    help="fixed scalar observation noise, standardised units")
    ap.add_argument("--raw2", action="store_true",
                    help="feed the 2 centred profiles instead of 4 sym/anti channels")
    ap.add_argument("--data", default="data/dataset.npz", help="prepared dataset .npz")
    ap.add_argument("--out", default="models", help="where to write weights and latents")
    a = ap.parse_args()
    for s in a.sigma:
        main(s, a.data, a.out, raw2=a.raw2)
