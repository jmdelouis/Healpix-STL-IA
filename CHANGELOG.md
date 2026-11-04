# Changelog

Every change of the code or of its behaviour is recorded here, newest first, together with the documentation pages it updates (see `CONTRIBUTING.md`).

## Unreleased

### Added
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
- The STL-Dev-MPI engine is imported when a training starts, so the package imports without it.
- `scripts/train_a100.slurm` is replaced by `scripts/train_gpu.slurm`, whose Slurm options come from the environment (`GPU_OPTS`).
