# micromatters-bvae

A β-VAE that learns the physiological identity of single *Escherichia coli* cells from one
snapshot: cell-cycle phase, growth rate, and the spatial organization of ribosomes and the
nucleoid, none of which the network is ever shown during training.

The model takes the mean-normalized RplA (ribosome) and HupA (nucleoid) linescans of a single
cell, together with its length and its two mean fluorescence concentrations, and compresses them
into eight latent coordinates. A random forest then reads physiological variables off those
coordinates. This is the network behind Figure 8 of the MICROMATTERS ERC Starting Grant proposal
(WP3.1).

## The result

Feeding the concentrations to the encoder is not enough to put them in the latent space. In
earlier versions the observation noise on the scalar inputs was learned, and the ELBO's cheapest
solution was to inflate that noise on three numbers and spend the capacity on the 400 profile
values instead — the latent simply discarded the concentrations.

Fixing the scalar noise instead of learning it is what changes the representation:

| model | scalar noise | phase | growth (frame) | growth (cycle) | log ratio from latent |
|---|---|---|---|---|---|
| L8 (length only) | — | 0.701 | 0.373 | 0.472 | 0.397 |
| L8c (+ concentrations) | learned | 0.704 | 0.387 | 0.474 | 0.410 |
| L8s01 (+ concentrations) | fixed, 0.1 | 0.778 | 0.491 | 0.559 | 0.827 |
| L8s03 (+ concentrations) | fixed, 0.3 | 0.710 | 0.403 | 0.507 | 0.437 |
| **L8s01raw2** (raw 2-channel) | **fixed, 0.1** | **0.793** | **0.513** | **0.550** | **0.941** |

Held-out R², random-forest read-out from the latent alone. `L8s01raw2` is the model used in the
proposal and the one whose weights are in `models/`. With a learned scalar noise the latent
recovers the RplA/HupA log ratio at R² = 0.41; with the noise pinned at 0.1 it recovers it at
0.94.

Full tables for four read-out variants, linear probes and the pole-flip symmetry analysis are in
[`results/results_v6.md`](results/results_v6.md) and
[`results/results_raw2.md`](results/results_raw2.md).

## Model

Convolutional β-VAE, 8 latent dimensions, β = 2.

- **Encoder** — three `Conv1d` layers (32, 64, 128 channels; kernel 5, stride 2, GELU) over the
  100-bin linescans, flattened and concatenated with the standardized scalars, then a 256-unit
  layer to μ and log σ².
- **Decoder** — 256 → 128 × 13 → three `ConvTranspose1d` layers back to 100 bins, plus a separate
  two-layer head that reconstructs the scalars.
- **Likelihood** — Gaussian with a learned per-channel variance on the profiles and a *fixed*
  variance on the scalars (σ_s = 0.1 in standardized units). This is the whole point: see above.
- **Inputs** — `L8s01raw2` uses the two mean-normalized profiles directly, centred by subtracting
  1. The `L8s01` variant instead splits each profile into its symmetric and antisymmetric halves,
  so that pole identity lives only in the antisymmetric channels.
- **Training** — 40 epochs, batch 512, Adam at 1e-3, seed 0.

## Data

The training set is 187,681 snapshots from 4,122 complete division cycles of *E. coli* CJW7323
(RplA-GFP, HupA-mCherry) growing in a mother machine, split **by trajectory** so that no cycle
appears on both sides: 159,835 training and 27,846 held-out snapshots, 3,504 and 618 cycles.

The prepared dataset is 250 MB and is not in this repository. `scripts/prepare_dataset.py` builds
it from the single-cell arrays of Papagiannakis *et al.*, *eLife* **14**, RP104276 (2025), BioImage Archive (S-BIAD1658); point
`--data-root` at that dataset and `--out` at where you want the `.npz`. It stores four channels —
the two linescans plus two nucleoid-position channels, one bump per segmented nucleoid and one at
their median — so that adding a real marker channel later (the planned ParB-BFP/*parS* *oriC*
reporter) is a drop-in. The β-VAE uses only channels 0 and 1.

Cell-cycle phase, growth rate and nucleoid position are carried through the pipeline but **never
used for training**. They exist only to test what the latent space recovers on its own.

## Usage

```bash
pip install -r requirements.txt

# 1. build dataset.npz from the source arrays (once)
python scripts/prepare_dataset.py --data-root /path/to/ERC_data --out data/dataset.npz

# 2. train
python -m bvae.train 0.1 --raw2 --data data/dataset.npz     # L8s01raw2, the proposal model
python -m bvae.train 0.1 0.3 --data data/dataset.npz        # the sym/anti variants

# 3. evaluate — writes results_v6.md
python -m bvae.evaluate --data data/dataset.npz --models models/

# 4. read-out from a named feature set
python -m bvae.features '["latent","length"]' --data data/dataset.npz
```

To use the trained model without retraining:

```python
import numpy as np
from bvae.model import load_model, scalar_inputs, prep_profiles

model, ck = load_model("models/vae_L8s01raw2.pt")
d = np.load("data/dataset.npz", allow_pickle=True)
X = prep_profiles(d["X"][:, [0, 1]], ck)
S = scalar_inputs(d, ck)
```

Or skip the encoder entirely: `models/latents_L8s01raw2.npz` holds the 8 latent means and
standard deviations for all 187,681 snapshots, in dataset order.


## Contents

```
bvae/model.py       network, ELBO, symmetry decomposition, checkpoint loading
bvae/train.py       training, one model per fixed scalar noise
bvae/evaluate.py    held-out R² tables across models and read-outs
bvae/features.py    R² for an arbitrary named feature set
scripts/            dataset preparation
models/             trained weights and latents for L8s01raw2
notebooks/          the annotated report and its figures
results/            result tables and training logs
```

## Citation

If you use this, please cite the dataset paper:

> Papagiannakis, A. *et al.* Nonequilibrium polysome dynamics promote chromosome segregation and
> its coupling to cell growth in *Escherichia coli*. *eLife* **14**, RP104276 (2025).

## License

MIT — see [LICENSE](LICENSE).
