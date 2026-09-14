# v6 results — concentrations as VAE inputs with fixed scalar noise

All 8-dim models, same split; held-out R² (RF read-out unless stated).

## Read-out Z: latent only

| model | phase | growth (frame) | growth (cycle-avg) | position | compaction | polysome asym. |
|---|---|---|---|---|---|---|
| L8 (length only) | 0.701 | 0.373 | 0.472 | 0.714 | 0.960 | 0.880 |
| L8c (+ conc., learned sigma_s) | 0.704 | 0.387 | 0.474 | 0.715 | 0.961 | 0.885 |
| L8s01 (+ conc., sigma_s = 0.1) — v6 | 0.778 | 0.491 | 0.559 | 0.695 | 0.949 | 0.831 |
| L8s03 (+ conc., sigma_s = 0.3) — v6 | 0.710 | 0.403 | 0.507 | 0.712 | 0.959 | 0.859 |
| L8s01raw2 (raw 2-ch linescans, sigma_s = 0.1) — v6 | 0.793 | 0.513 | 0.550 | 0.704 | 0.952 | 0.792 |

## Read-out B: latent + length

| model | phase | growth (frame) | growth (cycle-avg) | position | compaction | polysome asym. |
|---|---|---|---|---|---|---|
| L8 (length only) | 0.743 | 0.402 | 0.521 | 0.715 | 0.962 | 0.881 |
| L8c (+ conc., learned sigma_s) | 0.747 | 0.415 | 0.541 | 0.716 | 0.963 | 0.886 |
| L8s01 (+ conc., sigma_s = 0.1) — v6 | 0.821 | 0.499 | 0.575 | 0.691 | 0.949 | 0.831 |
| L8s03 (+ conc., sigma_s = 0.3) — v6 | 0.764 | 0.433 | 0.552 | 0.712 | 0.961 | 0.864 |
| L8s01raw2 (raw 2-ch linescans, sigma_s = 0.1) — v6 | 0.823 | 0.514 | 0.580 | 0.699 | 0.951 | 0.794 |

## Read-out A: latent + length + log RplA + log HupA

| model | phase | growth (frame) | growth (cycle-avg) | position | compaction | polysome asym. |
|---|---|---|---|---|---|---|
| L8 (length only) | 0.818 | 0.494 | 0.573 | 0.714 | 0.961 | 0.877 |
| L8c (+ conc., learned sigma_s) | 0.818 | 0.497 | 0.583 | 0.716 | 0.962 | 0.883 |
| L8s01 (+ conc., sigma_s = 0.1) — v6 | 0.829 | 0.499 | 0.579 | 0.691 | 0.947 | 0.836 |
| L8s03 (+ conc., sigma_s = 0.3) — v6 | 0.824 | 0.500 | 0.594 | 0.711 | 0.960 | 0.857 |
| L8s01raw2 (raw 2-ch linescans, sigma_s = 0.1) — v6 | 0.822 | 0.499 | 0.577 | 0.698 | 0.949 | 0.796 |

## Read-out C: latent + length + log ratio

| model | phase | growth (frame) | growth (cycle-avg) | position | compaction | polysome asym. |
|---|---|---|---|---|---|---|
| L8 (length only) | 0.820 | 0.508 | 0.569 | 0.714 | 0.961 | 0.880 |
| L8c (+ conc., learned sigma_s) | 0.820 | 0.509 | 0.579 | 0.715 | 0.962 | 0.884 |
| L8s01 (+ conc., sigma_s = 0.1) — v6 | 0.829 | 0.503 | 0.570 | 0.691 | 0.949 | 0.831 |
| L8s03 (+ conc., sigma_s = 0.3) — v6 | 0.825 | 0.517 | 0.574 | 0.711 | 0.961 | 0.862 |
| L8s01raw2 (raw 2-ch linescans, sigma_s = 0.1) — v6 | 0.824 | 0.501 | 0.581 | 0.698 | 0.951 | 0.793 |

## What the latent alone carries: linear probe vs RF, and the log ratio

| model | log ratio from latent (RF) | phase (linear) | growth (linear) | position (linear) | compaction (linear) |
|---|---|---|---|---|---|
| L8 (length only) | 0.397 | 0.553 | 0.168 | 0.603 | 0.628 |
| L8c (+ conc., learned sigma_s) | 0.410 | 0.567 | 0.209 | 0.616 | 0.649 |
| L8s01 (+ conc., sigma_s = 0.1) — v6 | 0.827 | 0.713 | 0.446 | 0.596 | 0.724 |
| L8s03 (+ conc., sigma_s = 0.3) — v6 | 0.437 | 0.598 | 0.275 | 0.600 | 0.671 |
| L8s01raw2 (raw 2-ch linescans, sigma_s = 0.1) — v6 | 0.941 | 0.743 | 0.465 | 0.594 | 0.740 |
