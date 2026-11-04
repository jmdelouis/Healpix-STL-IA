# Changelog

Every change of the code or of its behaviour is recorded here, newest first, together with the documentation pages it updates (see `CONTRIBUTING.md`).

## Unreleased

### Added
- **L-BFGS reference.** `stlia-train-denoise --optimizer lbfgs` uses no network: it minimises the training losses directly over the pixels of `u`, with the native L-BFGS of STL-Dev-MPI (`native_lbfgs`, strong-Wolfe line search). This is the per-map method of Delouis et al. (2022).
  - New options: `--lbfgs-rounds`, `--lbfgs-iter`, `--lbfgs-history`. The noise realisations are fixed within a round and redrawn between rounds.
  - The run `l256_dl123` is added to `scripts/reproduce.sh`.
  - Documentation updated: method (*The per-map reference*), user guide (*The L-BFGS reference*, outputs), `REPRODUCE.md`, and the paper (Sect. 2.3, "Per-map reference", and appendix).
- **Package.** The repository is now an installable package, `healpix-stl-ia`:
  - `pyproject.toml`, with commands `stlia-train-denoise` and `stlia-train-patches`;
  - the training programs move to `stlia.train_denoise` and `stlia.train_patches`, and `train_denoise.py` and `train_patches.py` stay as thin wrappers.
- **Documentation.** A Sphinx site, published on GitHub Pages:
  - installation, method, user guide, reproduction recipe;
  - command-line and API pages generated from the code.
- **Tests and CI.**
  - Unit tests: HEALPix geometry against healpy, the identity of the untrained network, gradients, absence of face edges, patches against the full sky, the loss, the command lines, and the engine gradient (when STL-Dev-MPI is installed).
  - Continuous integration: tests, documentation build with warnings as errors and deployment, and a check that every code change updates the documentation or this changelog.
- **Real-data mode.** `stlia-train-denoise --real` denoises the PR3 maps themselves: the full-mission and half-mission maps are the data. `scripts/make_figures_real.py` draws its figures.
- **GPU disc engine.** `stlia-train-patches` runs the disc statistics on the GPU with `--stl-backend gpu`, with the options `--cache-bytes` and `--reserve-bytes`.
- **Reproduction recipe.** `REPRODUCE.md` and machine-independent scripts: `scripts/env.sh`, `download_planck.sh`, `prepare_data.slurm`, `build_bridge.slurm`, `reproduce.sh`, `train_gpu.slurm`, `train_cpu.slurm`, `train_patches_gpu.slurm`.

### Changed
- `scripts/make_figures_real.py`: the full-sky figure shows the data `d` next to `u` and `d - u` (paper: figure caption updated).
- **Noise-bias wording.** The paper, the method page and the docstring of the L-BFGS reference now say that no noise bias is frozen: the statistics of `u + n_i` are recomputed at the current `u` at every evaluation. They said, wrongly, that the biases were re-estimated between batches of iterations as in Delouis et al. (2022).
- **Paper.** New section *Denoising the PR3 maps*: the network run `r256_dl123`, with the BICEP2 field, the EE/BB spectra against `d1 x d2` and the full-sky maps.
- The STL-Dev-MPI engine is imported when a training starts, so the package imports without it.
- `scripts/train_a100.slurm` is replaced by `scripts/train_gpu.slurm`, whose Slurm options come from the environment (`GPU_OPTS`).
