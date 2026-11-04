# Data

The training commands read NumPy maps from one directory (`--data`). These maps are:

- in the HEALPix NESTED order;
- in µK_CMB for the 353 GHz maps, and in MJy/sr for the 857 GHz map;
- degraded from the native $N_{\rm side}=2048$ by parent means, where every pixel is the mean of its four children (the pyramid of the network and of the engine).

## Download

`scripts/download_planck.sh` fetches the files of the paper from the Planck Legacy Archive, about 60 GB:

| file | use |
|---|---|
| `HFI_SkyMap_353-psb_2048_R3.01_full.fits` | sky (mock data) or observation $d$ (real data) |
| `HFI_SkyMap_353-psb_2048_R3.01_halfmission-{1,2}.fits` | half missions $d_1, d_2$ (`l1`) |
| `HFI_SkyMap_857_2048_R3.01_full.fits` | template $T$ (`l3`) |
| `ffp10_noise_353_psb_{full,hm1,hm2}_map_mc_000NN.fits` | FFP10 noise simulations, NN = 00 to 49 |

## Degraded maps

`scripts/prepare_planck.py` degrades one map, and `scripts/noise_std.py` degrades a set of simulations and computes their per-pixel standard deviation. `scripts/prepare_data.slurm` runs both on everything. The resulting names are:

| name | content |
|---|---|
| `planck353_{I,Q,U}_n{nside}.npy` | full-mission map |
| `planck353hm{1,2}_{I,Q,U}_n{nside}.npy` | half-mission maps (unseen pixels filled from the full mission) |
| `planck857_I_n{nside}.npy` | 857 GHz intensity |
| `planck353_noise_{I,Q,U}_n{nside}_mc{k}.npy` | FFP10 full-mission simulation `k` |
| `planck353_noisehm{1,2}_{I,Q,U}_n{nside}_mc{k}.npy` | the half-mission simulations of the same realisation |
| `planck353_noise_std_{I,Q,U}_n{nside}.npy` | per-pixel std of simulations 0–9 (`--noise pr3white`) |

The simulations are degraded *before* their standard deviation is taken, so the large-scale correlations of the PR3 noise set the level of a degraded pixel. At $N_{\rm side}=256$ the noise rms is 31.6 µK (Q) and 32.8 µK (U).

## Your own maps

Any NESTED map degraded the same way can be used. Give it the name `planck353_<field>_n<nside>.npy`, provide simulations of its noise with the names above, and select them with `--sims-train` and `--sims-test`. The `l1` term also needs two independent half-data maps, and `l3` needs a template.
