"""beta-VAE for multi-channel bacterial linescans.

Two input representations are supported through `channels`:

  * `channels=2` — the two mean-normalised profiles (RplA, HupA) centred by subtracting 1.
    This is the `L8s01raw2` model used in the proposal.
  * `channels=4` — each profile split into its symmetric half (x + flip(x))/2 and its
    antisymmetric half (x - flip(x))/2, fed as separate channels. Pole identity then lives
    only in the antisymmetric channels, so latent axes come out symmetry-pure by
    construction rather than as mixtures.

The observation noise is Gaussian with a learned per-channel variance on the profiles and a
separate variance on the scalar inputs. Pinning that scalar variance instead of learning it is
what forces the latent to carry the concentrations — see the README.
"""
import torch
import torch.nn as nn


class VAE(nn.Module):
    """zdim latent dims, `channels` linescan channels, `n_scalars` scalar inputs."""

    def __init__(self, zdim=8, channels=4, n_scalars=1):
        super().__init__()
        self.channels = channels
        self.n_scalars = n_scalars
        self.enc = nn.Sequential(
            nn.Conv1d(channels, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.GELU(),
            nn.Flatten(),
        )
        self.enc_fc = nn.Sequential(nn.Linear(128 * 13 + n_scalars, 256), nn.GELU())
        self.mu = nn.Linear(256, zdim)
        self.logvar = nn.Linear(256, zdim)
        self.dec_fc = nn.Sequential(
            nn.Linear(zdim, 256), nn.GELU(), nn.Linear(256, 128 * 13), nn.GELU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 5, stride=2, padding=2, output_padding=0),
            nn.GELU(),
            nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(32, channels, 5, stride=2, padding=2, output_padding=1),
        )
        self.dec_len = nn.Sequential(nn.Linear(zdim, 32), nn.GELU(),
                                     nn.Linear(32, n_scalars))
        self.log_sig2_x = nn.Parameter(torch.zeros(channels))
        self.log_sig2_l = nn.Parameter(torch.zeros(n_scalars))

    def encode(self, x, s):
        h = self.enc(x)
        if s.dim() == 1:
            s = s[:, None]
        h = self.enc_fc(torch.cat([h, s], dim=1))
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 13)
        shat = self.dec_len(z)
        return self.dec(h)[:, :, :100], (shat[:, 0] if self.n_scalars == 1 else shat)

    def forward(self, x, s):
        mu, logvar = self.encode(x, s)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        xhat, shat = self.decode(z)
        return xhat, shat, mu, logvar


def elbo_terms(m, x, s, xhat, shat, mu, logvar):
    """Gaussian NLL with per-channel and per-scalar variance, plus KL."""
    v2 = m.log_sig2_x.exp()[None, :, None]                  # (1, C, 1)
    nll_x = (0.5 * ((x - xhat) ** 2 / v2
                    + m.log_sig2_x[None, :, None])).sum((1, 2)).mean()
    if s.dim() == 1:
        s = s[:, None]
    if shat.dim() == 1:
        shat = shat[:, None]
    nll_s = (0.5 * ((s - shat) ** 2 / m.log_sig2_l.exp()[None, :]
                    + m.log_sig2_l[None, :])).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return nll_x, nll_s, kl


def sym_anti(Xc):
    """(N, 2, 100) centred profiles -> (N, 4, 100) symmetry-adapted channels.

    Channel order: RplA_sym, RplA_anti, HupA_sym, HupA_anti.
    Exact inverse: x = sym + anti.
    """
    import numpy as np
    flip = Xc[:, :, ::-1]
    sym = 0.5 * (Xc + flip)
    anti = 0.5 * (Xc - flip)
    return np.stack([sym[:, 0], anti[:, 0], sym[:, 1], anti[:, 1]],
                    axis=1).astype(np.float32)


def load_model(path):
    """Load a checkpoint (any latent width, 1 or 3 scalar inputs)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = VAE(ck["latent_dim"], channels=ck.get("channels", 4),
            n_scalars=ck.get("n_scalars", 1))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def scalar_inputs(d, ck):
    """(N, n_scalars) standardised scalar inputs matching the checkpoint:
    [length] or [length, log corrected mean RplA, log corrected mean HupA]."""
    import numpy as np
    if ck.get("n_scalars", 1) == 1:
        return ((d["length"] - ck["len_mu"]) / ck["len_sd"]
                ).astype(np.float32)[:, None]
    f = d["features"]
    fn = [str(x) for x in d["feature_names"]]
    is1m = np.array(["1minint" in str(t) for t in d["traj_ids"]])[d["traj_idx"]]
    log_r = np.log(f[:, fn.index("mean_fluor_2")] * np.where(is1m, 1.4106, 1.0))
    log_h = np.log(f[:, fn.index("mean_fluor_3")] * np.where(is1m, 1.5862, 1.0))
    if "log_ratio" in list(ck.get("scalar_names", [])):      # [length, log ratio]
        raw = np.c_[d["length"], log_r - log_h]
    else:                                                    # [length, log RplA, log HupA]
        raw = np.c_[d["length"], log_r, log_h]
    return ((raw - ck["scalar_mu"]) / ck["scalar_sd"]).astype(np.float32)


def prep_profiles(X2, ck):
    """(N, 2, 100) mean-normalised profiles -> network input for this checkpoint:
    4 sym/anti channels (ck['channels'] == 4) or the 2 centred profiles (== 2)."""
    import numpy as np
    if ck.get("channels", 4) == 4:
        return sym_anti(X2 - 1.0)
    return (X2 - 1.0).astype(np.float32)


def reassemble(x, ck):
    """decoded network output (N, C, 100) -> (N, 2, 100) measured-space profiles (mean 1)."""
    import numpy as np
    if ck.get("channels", 4) == 4:
        return np.stack([x[:, 0] + x[:, 1], x[:, 2] + x[:, 3]], 1) + 1.0
    return x + 1.0
