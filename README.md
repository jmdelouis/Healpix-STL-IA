# Healpix-STL-IA

Amortized scattering-based component separation on the sphere.

A HEALPix U-Net / autoencoder maps an observed map `d` to a clean map `u = f_θ(d)`.
It is trained without ground truth: the loss only compares scattering statistics
Φ computed with [STL-Dev-MPI](https://github.com/jmdelouis/STL-Dev-MPI)
(e.g. Φ(u + n_i) vs Φ(d)), following the denoising setup of Delouis et al. (2022)
but replacing the per-map L-BFGS optimisation by a trained network.

- `text/` : A&A paper (synchronised with Overleaf via `git subtree`)
- data and heavy computations live on the HPC (`$SCRATCH`), not in this repository

## Prototype (`train_denoise.py`)

- `stlia/unet.py`: U-Net on the 12 HEALPix base faces (NESTED pyramid = 2x2 pooling, face-aware padding), `u = d - r(d)` with `r` initialised to zero.
- `stlia/stl_torch.py`: STL-Dev-MPI scattering covariance (`stl-bridge` `NativeEngine`) as a torch autograd function, and the denoising loss
  `sum_c w_c |mean_i S_c(u + n_i) - S_c(d)|^2` (ergodic weights).
- `scripts/prepare_planck.py`: Planck PR3 353 GHz map -> NESTED I/Q/U `.npy` at nside 256/128/64 (run on the HPC).
- Demo protocol: white noise, train on Q (fresh noise each step), test on U.

```
source scripts/garoupe_env.sh
$PY train_denoise.py --data $G/data --out $G/runs/n64 --nside 64 --steps 3000
```
