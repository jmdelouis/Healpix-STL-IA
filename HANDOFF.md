# State of the project (hand-over, 10 November 2026)

This page is for whoever takes the project over: what exists, where it is, what is finished, what is
open, and what was tried and failed. The method and the results are in the paper (`text/main.tex`);
every run and figure can be rebuilt with `REPRODUCE.md`. Read those two first; this page only links
them together and adds what they do not say.

## 1. What the project is

A HEALPix U-Net that denoises the Planck PR3 353 GHz Q and U maps without clean training data. It is
trained with the scattering-statistics loss of Delouis et al. (2022): the statistics of u + noise must
match those of d, with noise biases frozen from FFP10 simulations. A spectrum term `L_cl` ties the EE/BB
spectra of u to the half-mission cross spectrum. The network, the statistics, their gradient and Adam
all run on one GPU, in the Rust engine of STL-Dev-MPI (`NativeUNet`, driver
`python/stl_ensemble/train_native_unet.py` there). This repository holds the PyTorch version
(`stlia/`), the scripts, the figures and the paper.

## 2. Where everything is

| what | where |
|---|---|
| this repository | `github.com/jmdelouis/Healpix-STL-IA`, branch `main` |
| the paper | `text/` here, also the Overleaf project kept in sync by `git subtree` (`CLAUDE.md`, rule "The paper") |
| the engine and the native trainer | `github.com/jmdelouis/STL-Dev-MPI`; the commits used are listed in the paper's reproduction appendix and in `REPRODUCE.md` Sect. 7.1. The newest is `9386f94` (below). |
| HPC (Datarmor, Ifremer) | `$HOME/stlia` = `R`: `src` (a copy of this repository), `data/planck` (downloads), `data/maps` (nside 256/128/64), `data/ext` (GNILC, BK14 mask), engine builds `stl_*`, older runs in `runs/` |
| HPC, large files | `/work/jmdeloui/stlia`: `maps_hi` (nside 512 and 1024, 55 GB), `sky_d12`, `sim_d12` (known sky), `gnilc` (nside 512), and the runs of the paper's latest maps in `runs/` (`n256_ebqu_c16_cl3n`, `sim_d12_ebqu_cl3n`, `n512_ebqu_c16_cl3n_g`, ...) |
| test GPU machine (garoupe, 3 T4) | `/tmp/jmdeloui-prof` and `/tmp/jmdeloui-stlia/data` (nside 64/128 data). `/tmp` is not kept: everything there is reproducible from the HPC data and the commits above. |

The paths are those of J.-M. Delouis's accounts; a new person needs their own copy (`REPRODUCE.md`
from the start). Python on the HPC: `/scratch/jmdeloui/stl-bench128/env/bin/python`.

## 3. What is finished

The paper (15 pages, A&A format) is complete except the first author (`\todo{first author}`; J.-M.
Delouis is last author). Its maps and numbers come from these runs:

| run | role | engine (`STL`) | check (`eval 3000`, Q / U err/noise) |
|---|---|---|---|
| `n256_ebqu_c16_cl3n` | **reference**, nside 256, `L_cl` to 3 nside - 1 | `stl_mem` (9d0a00f) | 0.866 / 0.789 |
| `sim_d12_ebqu_cl3n` | known-sky validation (PySM d12 + FFP10), same options | `stl_mem` | 0.902 / 0.834 |
| `n512_ebqu_c16_cl3n_g` | reference at nside 512, `L_cl` to 1535, `--spike-guard 3` | `stl_guard` (76d76eb) | 0.976 / 0.888 |
| `n256_ebqu_c16`, `n256_ebqu_c16cl`, `sim_d12_ebqu`, `n512_ebqu_c16` | the same with `L_cl` to 2 nside - 1 (comparison rows) | `stl_ebqu`, `stl_cl`, `stl_mem` | see `REPRODUCE.md` 7.3 |

The section "History of the tests and of the choices" of the paper lists every alternative tried, with
the result that decided it. `CHANGELOG.md` records every change.

## 4. Open problems

1. **nside 1024 never trained.** `n1024_ebqu_c16` (`--cl-lmax 3071`) was started twice and cancelled,
   waiting for the nside 512 check. It fits: 50.5 GB of GPU memory at the first live-bias step, which
   took 174 s, so the 200 live steps alone take about 10 h. Run it with `STL=stl_guard --spike-guard 3`.
   Expect the problems of point 2 at larger scale.
2. **u rises above the cross spectrum above l = 2 nside** (at 512: 2.5 times in EE and 7 in BB for
   1024 <= l < 1280, in all three 512 trainings). The cross spectrum is a few per cent of the noise there,
   and the removed part has 0.98-0.99 of the noise spectrum, so the excess is small in absolute terms. The
   cause is not understood. It decreased at nside 256 when `L_cl` went from 2 nside - 1 to 3 nside - 1.
3. **BB excess at intermediate scales.** BB of u is 1.15-1.22 times the cross spectrum for 450 <= l < 768
   at nside 256, and 1.3-1.6 for 450 <= l < 1024 at 512 (|b| > 30 deg). Over |b| > 60 deg, BB is 1.56 times
   it for 300 <= l < 450 at nside 256 (1.17 at 512). The per-map L-BFGS solution of the same loss has it
   too (1.38), so it comes from the loss, not from the network. The known-sky test does not reproduce it
   (BB 0.81 of the truth there), so it is not proven to be an artefact either.
4. **Large-scale residual at nside 512.** The removed part is 1.5 (EE) and 2.4 (BB) times the noise
   for 10 <= l < 30 over |b| > 30 deg, against 1.0 at nside 256.
5. **Adam spikes.** At nside 512 the loss jumps by a factor 3 to 100 at a few steps; without the guard the
   training does not recover (the first `cl3n` run at 512). The guard (restore the snapshot of the last
   50-step mark, reset Adam, halve the learning rate) fixes it; the cause of the spikes was not studied.
   The nside 256 runs did not need it and do not use it.
6. **Spread between trainings.** Two trainings with the same loss end on maps that differ by a large
   fraction of the noise; the paper quantifies it. Several seeds per configuration would be needed for
   differences of a few per cent.
7. **The learned kernels against wavelets** (do the first-layer kernels look like the scattering
   wavelets?) was planned and not started.

## 5. Tried after the paper's history list, and not kept

- **L-BFGS on the weights of the network** (nside 64, T4, 923 s per run, full reference loss): loss
  0.479 with Adam, 0.534 with L-BFGS after the live steps, 0.488 with 500 Adam steps then L-BFGS. L-BFGS
  does not shorten the training. Options `--optimizer lbfgs` / `adam+lbfgs` in STL-Dev-MPI `9386f94`.
  This test is in the paper's history list.
- **Coarse-to-fine start** (train at nside 64, start nside 128 from it: `--init`, and `--init-cascade`
  in `9386f94`): after 1000 steps at nside 128, the loss is 1.80-2.18 with the warm starts against
  1.42-1.54 from scratch, at the same pixel error (0.68-0.73 of the noise). No gain. Not in the paper.
- **Where time could be saved, not tested:** fewer Adam steps (at nside 64 the loss reaches within 4 % of
  its final value at half the steps; at nside 512 it went 9.54, 9.13, 8.99 at steps 2000, 2500, 3000), and
  a shorter live-bias warm-up (it costs 37 % of the training time at nside 64).

## 6. How the numbers of the text are made

- `make_figures_real.py RUN DATA OUT [--nside 512] [--gnilc DIR --field-mask MASK]`: the figures, the
  correlations with GNILC in the BICEP2 and ACTPol fields, and band powers (`--bcut 60` for |b| > 60 deg).
- `scripts/text_numbers.py RUN DATA NSIDE`: removed rms by latitude band, E/B and P parts, (d - u)/n
  and u/(d1 x d2) by multipole bin over |b| > 30 and 60 deg (Table 3 and the paragraphs around it).
- `eval_truth.py` and `scripts/mean_truth.py`: the known-sky table (means over the noises mc1-9).

## 7. Practical pitfalls

`REPRODUCE.md` ends with a list of them. In addition:
- On the HPC, `export R=$HOME/stlia` **before** `source scripts/hpc_env.sh`. Otherwise `R` defaults to
  `/scratch/$USER/healpix-stl-ia`, an incomplete older copy, and the scripts fail with "file not found".
- The login shell of garoupe is csh: send scripts with `ssh garoupe bash -s <<'EOF' ... EOF`.
- HPC GPU jobs: one A100 per run, at most two GPUs per user.
- Overleaf: check that Overleaf has no new commit (`git fetch overleaf`, compare `overleaf/main^{tree}` with
  `HEAD:text`) before `git subtree split --prefix text` and the push; never `subtree pull`.
