# Changelog

Every change of the code or of its behaviour is recorded here, newest first, together with the documentation pages it updates (see `CONTRIBUTING.md`).

## Unreleased

### Fixed
- **Run directory.** The scripts write the runs and the Slurm logs to `$RUNS` (default `$R/runs`), so that they can go to another file system when the work one is full. Documentation updated: `REPRODUCE.md` (environment, pitfalls).
- **Checkpoints.** `stlia-train-denoise` writes `model.pt` to a temporary file and renames it. A failed write, such as a full file system, now leaves the previous checkpoint intact instead of a truncated file. No documentation change: the outputs are the same.

### Changed
- **High-latitude spectrum mask.** `l256_fz_cl`, `r256_joint_eebb` and `r256_fz_cl857` in `scripts/reproduce.sh` use `--cl-masks 0,30,45,60`, and the STL-Dev-MPI build of `$R/stl_sp2`: with masks up to 45 degrees only, the spectra of u at |b| > 60 deg were the least constrained. No change of code.
- **Frozen biases only where there is a noise bias.** With `--bias frozen`, the bias is zero for the coefficients whose noise average is the noise-free statistic: the noise enters them only linearly. These are the mean of `data`, `S2` of `l1`, `l2` and `l3`, and the `S3` of `l2`/`l3` with the noise-free map under the modulus (`unbiased()`). `--bias-all` restores the previous estimate for every coefficient, as used by `l256_fz`, whose line in `scripts/reproduce.sh` now has it. Documentation updated: method (*Which coefficients have a noise bias*), paper Sect. 2.4 (Table of biased coefficients).
- **No noise bias on the spectrum loss.** The model of `cl` is now `C_b(u, u)`, with live and frozen biases alike. The half-mission noises are independent, so `<C_b(u + n1, u + n2)> = C_b(u, u)`. Before, the live model averaged noise draws, which only added scatter to the gradient. The frozen model subtracted an estimated bias, close to zero but noisy, which could make the solution drift. Documentation updated: method (*Spectrum loss*), paper Sect. 2.4.

### Added
- **Native U-Net run.** `scripts/train_native.slurm` trains the native U-Net with the driver of STL-Dev-MPI (b580210, same outputs as `train_denoise`). The run `r256_native`, as `r256_fz`, is added to `scripts/reproduce.sh` and `REPRODUCE.md`.
- **Native U-Net benchmark.** `scripts/bench_native.slurm` times one frozen-bias step of the Rust U-Net of STL-Dev-MPI (`NativeUNet`, branch `claude/native-unet`) against the torch pipeline, at nside 64 and 256. Documentation updated: `REPRODUCE.md`.
- **Joint (Q, U) network.** `UNet(maps=2)` and `stlia-train-denoise --joint` (real data, Adam): one network denoises Q and U together and every step trains both. The run `r256_joint_eebb` is added to `scripts/reproduce.sh` and `REPRODUCE.md`. `--cl-spin 2` puts the spectrum loss on the EE and BB cross spectra (`BinnedSpectra(spin=2)`), through the spin-2 `SpectrumEngine` of STL-Dev-MPI 92f2aef. The new tests cover the joint network and EE/BB against healpy. Documentation updated: method (*A joint (Q, U) network*), user guide (full sky).
- **Spectrum loss in the engine.** `BinnedSpectra(backend="stl")` and `stlia-train-denoise --cl-backend {auto,torch,stl}` compute the pseudo-spectra and their gradient with the `SpectrumEngine` of STL-Dev-MPI (commit 5648ba1, branch `claude/spectrum`). They agree with the `torch` transforms to 1e-12, and are 16 times faster on a T4 at nside 256. `auto`, the default, falls back to `torch` with an older engine. A test compares both backends. Documentation updated: method (*Spectrum loss*), installation, `REPRODUCE.md`.
- **Runs `r256_cl857` and `l256_cl`.** The spectrum loss with live noise biases, for the network (with the 857 GHz conditioning) and for the L-BFGS reference. Its model has no noise bias in either scheme, since the half-mission noise draws are independent. Documentation updated: `REPRODUCE.md`.
- **Masked statistics.** `--stat-masks` averages every scattering coefficient of the terms under galactic masks, each with its ergodic weight. It uses the masked averages of the STL-Dev-MPI engine (commit 9038114, `NativeEngine(masks=...)`), and `DenoiseLoss` gains the `masks` and `npix` arguments.
  - The runs `r256_fz_cl857_mk` and `l256_fz_mk` are added.
  - Documentation updated: method (*Masked statistics*), user guide, installation, `REPRODUCE.md`, paper Sect. 2.4.
- **Frozen noise biases.** `--bias frozen` implements the scheme of Delouis et al. (2022): the bias of each term is estimated at a fixed `u` with `--bias-realizations` draws and kept constant (re-estimated every L-BFGS round, or every `--bias-every` network steps), and the model is `Phi(u) + B`. The default, `--bias live`, keeps the previous behaviour.
- **Spectrum loss `cl`.** A binned half-mission cross spectrum under galactic masks (`--cl-masks`, `--cl-lmin`, `--cl-lmax`, `--cl-width`, `--cl-apod`). It comes with the modules `stlia.sht` and `stlia.spectra`: a differentiable ring-FFT spherical harmonic transform and `anafast`, adapted from healpix-analyse and FOSCAT and tested against healpy, and binned masked spectra. The optional extra `spin` covers the E/B transforms.
- **Conditioning maps.** `UNet(cond=...)` and `stlia-train-denoise --cond log857`: the network receives log I_857 at its input and at every decoder level.
- **Runs.** `l256_fz`, `r256_fz`, `r256_fz_cl857` and `l256_fz_cl` are added to `scripts/reproduce.sh` and `REPRODUCE.md`.
- **Documentation.** Updated: method (noise biases, spectrum loss, conditioning), user guide, installation, API. Paper: Sect. 2.4 and the conclusion of Sect. 5.
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
- **Paper.** Sect. 5 adds the frozen-bias L-BFGS reference `l256_fz` and a table comparing the network and the two L-BFGS references on the PR3 maps.
- **Paper.** Sect. 5 now gives the per-map L-BFGS reference `l256_dl123` (Fig. 11), with its maps and spectra compared to those of the network.
- `scripts/make_figures_real.py`: the full-sky figure shows the data `d` next to `u` and `d - u` (paper: figure caption updated).
- **Noise-bias wording.** The paper, the method page and the docstring of the L-BFGS reference now say that no noise bias is frozen: the statistics of `u + n_i` are recomputed at the current `u` at every evaluation. They said, wrongly, that the biases were re-estimated between batches of iterations as in Delouis et al. (2022).
- **Paper.** New section *Denoising the PR3 maps*: the network run `r256_dl123`, with the BICEP2 field, the EE/BB spectra against `d1 x d2` and the full-sky maps.
- The STL-Dev-MPI engine is imported when a training starts, so the package imports without it.
- `scripts/train_a100.slurm` is replaced by `scripts/train_gpu.slurm`, whose Slurm options come from the environment (`GPU_OPTS`).
