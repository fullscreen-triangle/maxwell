# Contact-ladder validation on RepoRT

Analysis behind `docs/thermodynamics/chromatography-contact-ladder/chromatography-contact-ladder.tex`.

## Data

`sh fetch_data.sh` makes a sparse clone of [RepoRT](https://github.com/michaelwitting/RepoRT) into `data/RepoRT`.
RepoRT is licensed CC-BY-SA 4.0 (Kretschmer et al., *Nat Methods* 21:153–155, 2024).
The analysis used commit `420b550`.

## Run order

Run from the `analysis/` folder, with numpy, scipy, pandas and matplotlib installed.

| Step | Script | Output | Runtime |
|---|---|---|---|
| Dataset table | `build.py` | `repo.pkl` | about 1 min |
| Flow-rate test (prediction P1) | `v1_flow.py` | `v1_results.json`, `v1_fits.npz`, `v1_coarse.npy` | about 10 min, cached afterwards |
| Retained / unretained split | `v1_split.py` | `v1_split.json` | about 1 min |
| Dwell-volume sensitivity | `v1_vd.py` | `v1_vd.json` | a few minutes |
| Elution compositions | `v2_build.py` | `v2_phi.pkl` | about 1 min |
| Contacts versus carrier (P2, P3) | `v2_pairs.py`, then `v2b.py` | `v2_pairs.pkl`, `v2_results.json`, `v2b_results.json` | about 15 min |
| Hydrophobic-subtraction transfer (P4) | `v3_hsm.py` | `v3_results.json`, `v3_forward_*.csv`, `v3_inverse_*.csv` | about 15 min |
| Figures | `fig12.py`, `fig3.py` | written to the manuscript's `figures/` folder | seconds |

`ladder_gradient.py` holds the gradient-ladder integrator. It solves dξ/dt = 1/(t0(1+k)) with ln k = a − S(φ − 0.5), where φ is the organic fraction that has reached the analyte.
