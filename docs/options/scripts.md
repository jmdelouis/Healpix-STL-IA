# Scripts: data, evaluation, figures

The scripts of `scripts/` prepare the data, apply a trained network, measure it and draw the figures of the paper.
Run them with the Python of the environment (`$PY`, see {doc}`../installation`). The shell scripts read the environment
of `scripts/env.sh`.

## Environment and jobs

| script | role |
|---|---|
| `env.sh` | sets the environment from `R` (work directory): `RUNS`, `STL`, `G=$R/$STL`, `STL_NATIVE_LIBRARY`, `PYTHONPATH`, `PY`, `GPU_OPTS`, `CPU_OPTS`. Source it first: `export R=...; source $R/src/scripts/env.sh`. |
| `hpc_env.sh`, `site/datarmor.sh`, `garoupe_env.sh` | the settings of the machines used for the paper, as examples of a site file (`SITE_ENV`). On Datarmor, `export R=$HOME/stlia` **before** sourcing `hpc_env.sh`, whose default `R` is an older copy. |
| `build_bridge.slurm` | builds the STL-Dev-MPI engine of `$R/$STL/src` (`cargo build --release -p stl-bridge --features gpu`), 10 min |
| `train_native.slurm RUN [options]` | one run of the production trainer ({doc}`native-trainer`) with `--data $R/data/maps --out $RUNS/RUN`; `STL` selects the engine build. Give `--data` again to override. |
| `reproduce.sh sky\|runs\|apply\|figures` | every step of the paper (see {doc}`../reproduce`) |
| `train_gpu.slurm`, `train_cpu.slurm`, `train_patches_gpu.slurm` | runs of the PyTorch trainers ({doc}`torch-cli`) |

## Data

| script | usage | what it writes |
|---|---|---|
| `download_planck.sh [NSIMS]` | `bash scripts/download_planck.sh 50` | the PR3 353 GHz PSB full and half-mission maps, the 857 GHz map, and the FFP10 noise simulations 0..NSIMS-1 of the three 353 GHz maps, into `$R/data/planck` (60 GB for 50) |
| `download_ext.sh` | `bash scripts/download_ext.sh` | the GNILC dust map and the BK14 field mask, into `$R/data/ext` (1.6 GB) |
| `prepare_data.slurm` | `sbatch $CPU_OPTS scripts/prepare_data.slurm` | everything the trainers read at $\nside=256$, 128 and 64, in `$R/data/maps` (3 h on 4 cores, 15 GB) |
| `prepare_data_hires.slurm [OUT]` | `sbatch $CPU_OPTS scripts/prepare_data_hires.slurm` | the same at 512 and 1024, in `$R/data/maps_hi` (55 GB) |
| `prepare_planck.py IN.fits OUT` | `--name planck353`, `--fields IQU`, `--scale 1e6` (K to $\mu$K; 1 for 857 GHz), `--nside 256 128 64`, `--fill-from NAME` (pixels with no seen child taken from an already prepared map, for the half missions) | `OUT/{name}_{field}_n{nside}.npy`: NESTED parent means |
| `noise_std.py OUT SIM.fits...` | `--name noise` (or `noisehm1`, `noisehm2`), `--nside 256 128 64` | each simulation degraded as the sky, `planck353_{name}_{I,Q,U}_n{nside}_mc{k}.npy`, and their per-pixel std |
| `prepare_gnilc.py GNILC.fits OUT` | `--nside 256` | `gnilc353_{Q,U}_n{nside}.npy`, for the figures only |

## Known sky (validation)

| script | usage | what it does |
|---|---|---|
| `make_pysm_sky.py OUT` | `--model d12`, `--nside-in 2048`, `--nside 256 [512 ...]`, `--fwhm 4.94` (arcmin, 353 GHz beam) | evaluates a PySM 3 dust model at 353 and 857 GHz, smoothed by the beam: `sky353_{I,Q,U}_n{nside}.npy`, `sky857_I_n{nside}.npy`. Needs `pysm3` (`$PYSM_PY`). |
| `make_sim_obs.py SKY DATA OUT` | `--nside 256`, `--mc 0` (FFP10 simulation of the observation, outside the training ones), `--match 30` (deg), `--lrange 30 450` | gives the sky the EE and BB spectra of the PR3 cross spectrum over `--lrange` at $\vert b\vert>$ `--match` (a power law in $\ell$), adds the noise `mc`, and writes the files of a PR3 observation plus `truth353_{Q,U}`, with links to the noise files of `DATA` |
| `apply_native.py RUN SIM OUT.npz` | `--mcs 1-9`, and the options of the training: `--nside 256`, `--channels 16,32,48,64,64,64`, `--cond log857+sigma`, `--sims 10-49`, `--pol ebqu` (or `qu`), `--lib` (default `$G/src/python/stl_ensemble`), `--device`, `--backend` | rebuilds the network of `RUN/params.npy` and applies it to truth + noise `k` for each `k` of `--mcs`: `OUT_mc{k}.npz` (`Q`, `U`, `mc`). The options must be those of the training. |
| `eval_truth.py SIM DATA [RUN...]` | `--maps u_mc*.npz`, `--nside 256`, `--mc 0` | for each run or map: rms of $u-s$ over the noise by band of $\vert b\vert$, and over $\vert b\vert>30^\circ$ and $60^\circ$ the EE/BB ratios $C(u)/C(s)$, transfer, correlation and $C(u-s)/C(n)$ by band of $\ell$ |
| `mean_truth.py FILE` | `python scripts/mean_truth.py RUN/eval_truth.txt` | the means over the noises mc1-9 of the `eval_truth.py` lines (paper Table 2) |

## Measures and figures on the PR3 maps

| script | usage | what it does |
|---|---|---|
| `make_figures_real.py RUN DATA OUT` | `--nside 256`, `--sims 0-9` (evaluation noise), `--bcut 30` (latitude cut of the spectra), `--apod 5`, `--lmax` (default $3\nside-1$), `--gnilc DIR`, `--field-mask bk14_mask_gal_n0512.fits` | `real_maps.pdf` (full sky), `real_bicep.pdf`, `real_actpol.pdf` (fields with $d$, $u$, GNILC, $d-u$, $(d_1-d_2)/2$), `real_spectra.pdf` (EE, BB of $d$, $u$, $d_1\times d_2$, $d-u$, $(d_1-d_2)/2$, noise). Prints the correlation and rms of $u$ and $d$ against GNILC in each field, and band powers. |
| `text_numbers.py RUN DATA [NSIDE]` | `python scripts/text_numbers.py $RUNS/n256_ebqu_c16_cl3n $R/data/maps 256` | removed rms over noise, full sky and by band of $\vert b\vert$; its part in the E/B maps and in $P$; $(d-u)/n$ and $u/(d_1\times d_2)$ for EE and BB by bin of $\ell$ over $\vert b\vert>30^\circ$ and $60^\circ$ (paper Table 3 and Sect. 5) |
| `make_figures_sim.py SIM RUN OUT` | `--nside 256`, `--mc 0`, `--apply 'RUN/apply/u_mc*.npz'`, `--apod 5`, `--lmax` | `sim_maps.pdf` and `sim_spectra.pdf` of the known-sky validation |
| `make_figures.py`, `make_figures256.py`, `make_figure_highlat.py` | see their docstrings | figures of the early tests (mock data, PyTorch runs) |
