# Healpix-STL-IA

Amortized scattering-based component separation on the sphere.

A HEALPix U-Net / autoencoder maps an observed map `d` to a clean map `u = f_θ(d)`.
It is trained without ground truth: the loss only compares scattering statistics
Φ computed with [STL-Dev-MPI](https://github.com/jmdelouis/STL-Dev-MPI)
(e.g. Φ(u + n_i) vs Φ(d)), following the denoising setup of Delouis et al. (2022)
but replacing the per-map L-BFGS optimisation by a trained network.

- `text/` : A&A paper (synchronised with Overleaf via `git subtree`)
- data and heavy computations live on the HPC (`$SCRATCH`), not in this repository
