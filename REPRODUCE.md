# Reproducing the runs of the paper

This recipe rebuilds every run, figure and table of `text/main.tex` from
public data on any Linux machine with an NVIDIA GPU. It does not depend on
the cluster or on the person (or agent) running it. Each step lists:

- its command;
- its cost;
- a check, so that a failure is caught at the step that causes it.

Shell commands are bash. Do not run them from csh: on a csh login, pipe
the scripts to `bash -s` or start `bash` first. Files must keep Unix line
endings (LF).

## 0. What you need

| | used for the paper | minimum |
|---|---|---|
| GPU | NVIDIA A100 80 GB (Datarmor, Ifremer); T4 16 GB for tests | native U-Net: 16 GB for nside 256, 48 GB for 512, 80 GB for 1024; PyTorch runs: 40 GB for nside 256 |
| CPU node | 32 cores, 128 GB (CPU disc engine, `p256_l123`) | same; or use the GPU disc engine |
| disk | 150 GB (60 GB downloads, 15 GB maps, 55 GB maps at nside 512/1024, 2 GB external maps and known sky, 10 GB runs) | |
| Rust | 1.99.1 (rustup, stable) | stable toolchain, edition 2021 |
| CUDA | driver 13.0 + NVRTC 13.0 (A100); driver 11.4 + NVRTC 11.8 wheel (T4) | driver >= 11.4, a matching `libnvrtc` on `LD_LIBRARY_PATH` |
| Python | 3.11.18, torch 2.11.0+cu128, numpy 2.4.6, scipy 1.17.1, healpy 1.20.1, matplotlib 3.11.2 | 3.10+, torch with CUDA |
| Slurm | optional: without it every job runs with `bash` | |

Code:

- **This repository**, [`jmdelouis/Healpix-STL-IA`](https://github.com/jmdelouis/Healpix-STL-IA) (public).
- **[STL-Dev-MPI](https://github.com/jmdelouis/STL-Dev-MPI)** (public), at commit `20329e1d2aeb6b822e194f671367e9eed8eb8193`.
  - It provides the scattering statistics: the Rust engine `stl-bridge` and its Python wrapper `stl_port.native`.
  - The full-sky runs of Sect. 4 used its parent commit `66975a4dead97b8828721b6fbb67b2147a3fed48`. The full-sky statistics are identical in both commits.
  - `20329e1` adds the GPU disc engine and stops reading disc interpolation cells of weight <= 1e-9, a change at the rounding level.
  - `9038114` adds the masked statistics of the runs `*_mk`.
  - `5648ba1` (built on `9038114`) adds `SpectrumEngine`, a differentiable `anafast`. With it, `--cl-backend auto` computes the spectrum loss in the engine. Without it, the same loss comes from the transforms of `stlia.sht`; the two agree to 1e-12.
  - The nside 64 runs of Sect. 3 used an earlier build of the same full-sky statistics (an earlier commit of STL-Dev-MPI).

## 1. Work directory and environment

Everything lives under one work directory `$R`:

```
$R/src            this repository
$R/stl/src        STL-Dev-MPI checkout     ($R/$STL/src; STL defaults to "stl")
$R/stl/target     its build
$R/data/planck    PLA downloads
$R/data/maps      degraded maps (.npy)
$R/runs           one directory per run, figures in runs/figs*
```

```bash
export R=/path/to/workdir
mkdir -p $R/runs
git clone https://github.com/jmdelouis/Healpix-STL-IA $R/src
git clone https://github.com/jmdelouis/STL-Dev-MPI $R/stl/src
git -C $R/stl/src checkout 20329e1d2aeb6b822e194f671367e9eed8eb8193
# masked statistics (runs *_mk): a second checkout at 9038114 in $R/stl_mk/src, built with STL=stl_mk
# spectrum loss in the engine: a checkout at 5648ba1 (it also has the masked statistics), built with STL=stl_sp
# EE/BB spectrum loss of the joint (Q, U) network: a checkout at 92f2aef, built with STL=stl_sp2
# local spin-2 statistics (--pol-stats chi, run r256_chi): a checkout at f4cf030, STL=stl_spin
```

`scripts/env.sh` sets the environment from a few variables:

| variable | meaning |
|---|---|
| `R` | the work directory (required) |
| `SITE_ENV` | a file sourced first: modules, rustup, the CUDA runtime |
| `PY` | the Python interpreter (default `python3`) |
| `RUNS` | where the runs and the Slurm logs go (default `$R/runs`) |
| `CUDARC_CUDA_VERSION` | driver CUDA version, e.g. `13000`, `12080`, `11040`; read it from `nvidia-smi` |
| `GPU_OPTS`, `CPU_OPTS` | `sbatch` options of a 1-GPU job and of a 32-core job; leave empty to run with `bash` |

Write these lines once for your machine. `scripts/hpc_env.sh` and
`scripts/site/datarmor.sh` are the settings of the paper, given as an example:

```bash
export R=... SITE_ENV=... PY=... CUDARC_CUDA_VERSION=...
export GPU_OPTS="-p <gpu partition> --gres=gpu:1 -c 6 --mem=48G -t 12:00:00"
export CPU_OPTS="-p <cpu partition> -c 32 --mem=128G -t 24:00:00"
source $R/src/scripts/env.sh
```

**Check:**

```bash
$PY -c "import torch, healpy, stl_port.native; print(torch.cuda.is_available())"
```

It must print `True` on a GPU node.

## 2. Build the scattering engine

```bash
cd $R/src
sbatch $CPU_OPTS -o $R/runs/slurm.%j.log scripts/build_bridge.slurm   # or: bash scripts/build_bridge.slurm
```

The build takes about 10 min. It needs no MPI and no libclang.

**Check:** `$STL_NATIVE_LIBRARY` exists. On a GPU node, the following prints the number of coefficients, `2522`:

```bash
$PY -c "from stl_port.native import NativeEngine as E; print(len(E(64, 5, batch=1, backend='gpu').plan))"
```

## 3. Data

### Download

```bash
bash scripts/download_planck.sh 50
```

This takes 1 to 3 h and fills 60 GB, from the Planck Legacy Archive (`http://pla.esac.esa.int/pla/aio/product-action?...`). It fetches:

- `HFI_SkyMap_353-psb_2048_R3.01_{full,halfmission-1,halfmission-2}.fits`;
- `HFI_SkyMap_857_2048_R3.01_full.fits`;
- the FFP10 noise simulations `ffp10_noise_353_psb_{full,hm1,hm2}_map_mc_000NN.fits`, NN = 00 to 49.

**Check:** `$R/data/planck` holds 4 maps, `noise/` holds 50 files and `noise_hm/` holds 100.

### Preparation

```bash
sbatch $CPU_OPTS -o $R/runs/slurm.%j.log scripts/prepare_data.slurm   # or: bash scripts/prepare_data.slurm
```

This takes about 3 h on 4 cores, with 24 GB of memory. It degrades every map to nside 256, 128 and 64:

- NESTED pixel order, parent means;
- 353 GHz maps in µK_CMB, the 857 GHz map in MJy/sr.

**Check:**
- `$R/data/maps` holds about 1400 `.npy` files.
- The first line of every nside 256 run log (step 5) reads `noise pr3sims rms Q 31.61, U 32.78 uK (train std 262.9, test std 155.7)`.

## 4. Smoke test (5 min on one GPU)

```bash
bash scripts/train_gpu.slurm smoke --nside 64 --noise uniform --loss data=1 --steps 50 --eval-every 50 --log-every 10
tail -3 $R/runs/smoke/log
```

**Check:**
- The loss decreases.
- A step takes 0.2 s or less on an A100 or a T4.
- `eval 50` lines are printed for Q and U.

## 5. The runs

```bash
bash scripts/reproduce.sh runs
```

This submits the 20 runs listed in the script, which are independent of one another. Without Slurm, they run one after the other. Each run writes to `$R/runs/<name>/`:

- `log`;
- `train.json`, which holds the arguments, the loss history and the evaluations;
- `model.pt`;
- `eval_{Q,U}.npz`, which holds the maps.

| run | paper | job | time (A100) |
|---|---|---|---|
| `sph_base`, `sph_cosmo`, `sph_pr3`, `sph_pr3x4`, `n64_base` | Sect. 3, Table 1, Figs. 1-6 | GPU | 10 min each |
| `n256_data` | Sect. 4, Table 2 | GPU | 50 min |
| `n256_delouis` | Sect. 4, Table 2 | GPU | 2 h 20 |
| `p256_l123` | Sect. 4, Table 2 | 32 CPU cores | 2 h 50 |
| `r256_dl123` | the PR3 maps denoised | GPU | 2 h 50 |
| `l256_dl123` | the PR3 maps denoised without a network: the same losses minimised over the pixels by the L-BFGS of STL-Dev-MPI (reference) | GPU | about 1.5 h (estimate) |
| `r256_fz` | network, frozen noise biases | GPU | not yet measured |
| `r256_fz_cl857` | network, frozen biases, spectrum loss `cl` and 857 GHz conditioning | GPU | not yet measured |
| `l256_fz_cl` | L-BFGS reference, frozen biases and spectrum loss (masks up to 60 deg) | GPU | 1 h 09 |
| `r256_cl857` | network, live biases, spectrum loss `cl` and 857 GHz conditioning | GPU | not yet measured |
| `l256_cl` | L-BFGS reference, live biases and spectrum loss | GPU | not yet measured |
| `r256_native` | as `r256_fz`, with the native U-Net of STL-Dev-MPI (Rust, `NativeUNet`, b580210 built in `$R/stl_nu`; `scripts/train_native.slurm`) | GPU | 51 min |
| `r256_native_s{1,2,3}` | as `r256_native`, seeds 1-3 (network initialisation and noise draws): the spread between trainings | GPU | about 51 min each |
| `r256_joint_eebb` | one network for (Q, U) (`--joint`), live biases, EE/BB spectrum loss (`--cl-spin 2`, STL-Dev-MPI 92f2aef built in `$R/stl_sp2`) and 857 GHz conditioning | GPU | not yet measured |
| `r256_fz_cl857_mk` | as `r256_fz_cl857`, with the scattering statistics also under the masks 0, 20, 30, 45° (`--stat-masks`, STL-Dev-MPI 9038114 built in `$R/stl_mk`) | GPU | 3 h 35 |
| `r256_sig_wu` | as `r256_fz_cl857_mk`, with the noise level as a second conditioning map (`--cond log857+sigma`), 500 steps of live biases first (`--bias-warmup 500`) and the spectrum mask at 60° (`$R/stl_sp2`) | GPU | not yet measured |
| `r1k_ref` | as `r256_sig_wu` with 1000 steps and 200 warm-up steps (`$R/stl_sp2`) | GPU | not yet measured |
| `r1k_res` | as `r1k_ref`, with `l2` replaced by `res` (Phi(d - u) against the half-mission difference) and the spectrum terms `tcl` and `rcl` | GPU | not yet measured |
| `r256_eb` | as `r256_sig_wu`, one (Q, U) network with every term on the E and B maps (`--joint --pol-stats eb`, spin-2 transforms of `stlia.sht`; about 14 GB more GPU memory) | GPU | not yet measured |
| `r256_chi` | as `r256_sig_wu`, one (Q, U) network with every scattering term on chi_E + i chi_B = conj-eth^2 (Q + iU) (`--joint --pol-stats chi`, spin-2 engine of STL-Dev-MPI f4cf030 built in `$R/stl_spin`) and the spectrum loss on EE and BB (`--cl-spin 2`) | GPU | not yet measured |
| `l256_fz_mk` | L-BFGS reference, frozen biases, masked statistics | GPU | not yet measured |

`p256_l123` can instead use the GPU disc engine, with the same options:

```bash
sbatch $GPU_OPTS -o $R/runs/slurm.%j.log scripts/train_patches_gpu.slurm p256_gpu [options of p256_l123]
```

Set `CACHE_BYTES=3e9` on a 16 GB GPU. The test on a T4 ran at 2 s per step; its final full-sky evaluation does not fit in 16 GB.

**Check (final `eval` line of each log, or `train.json`):** the runs are not bit-reproducible on GPUs (atomic sums), so compare with tolerances:
- about 0.02 on the pixel error over the noise;
- about 20 % on each family of statistics.

The reference values are those of Tables 1 and 2 of the paper. For `n256_data`, for example:

- field Q: pixel error 1.09, S1 0.94 (d: 25);
- field U: pixel error 0.93, S1 2.0 (d: 27).

### Native U-Net (STL-Dev-MPI, Rust)

The U-Net of `stlia` also exists in Rust in STL-Dev-MPI (`NativeUNet`, commit `b580210`), trained with everything on one GPU. Build that commit in `$R/stl_nu` (`STL=stl_nu`, as above). The run `r256_native` trains it as `r256_fz` (`scripts/train_native.slurm`, driver `train_native_unet.py` of STL-Dev-MPI). To time one frozen-bias step against the torch pipeline at nside 64 and 256:

```bash
STL=stl_nu sbatch $GPU_OPTS -o $RUNS/slurm.%j.log scripts/bench_native.slurm   # -> $RUNS/bench_native/*.json
```

## 6. Figures and tables

```bash
bash scripts/reproduce.sh figures
```

This writes `$R/runs/figs64` (Figs. 1-6 and `results_table.tex`), `$R/runs/figs256` (Figs. 7-8 and `results256_table.tex`) and `$R/runs/figs_real`. Copy them to `text/figures/` and compile `text/main.tex` with `latexmk -pdf`.

## 7. The maps of the paper: known sky, reference, nside 512 and 1024

Sections 1-6 rebuild the history of the tests. The maps shown in the paper come from the native U-Net of
STL-Dev-MPI (driver `train_native_unet.py`, `scripts/train_native.slurm`), at three commits:

| `STL` | commit | runs |
|---|---|---|
| `stl_ebqu` | `596e3e9c7ef4c9af5007266dde8442f3e58e7fa8` | `n256_ebqu_c16`, the reference at nside 256 |
| `stl_cl` | `9be7aa0a7e03bae80cdba6db841c4e0c47a1d694` | `n256_ebqu_c16cl` (spectrum term in the trainer), `sim_d12_ebqu` (known sky) |
| `stl_mem` | `9d0a00fc8f476f70f64b6324d99cb0c334ef93a4` | `n512_ebqu_c16`, `n1024_ebqu_c16` (memory plan, host noise pools, `--mem-trace`) |

### 7.1 Engines

```bash
for b in stl_ebqu:596e3e9c7ef4c9af5007266dde8442f3e58e7fa8 stl_cl:9be7aa0a7e03bae80cdba6db841c4e0c47a1d694 \
         stl_mem:9d0a00fc8f476f70f64b6324d99cb0c334ef93a4; do
    git clone https://github.com/jmdelouis/STL-Dev-MPI $R/${b%%:*}/src
    git -C $R/${b%%:*}/src checkout ${b##*:}
    STL=${b%%:*} sbatch $CPU_OPTS -o $RUNS/slurm.%j.log scripts/build_bridge.slurm
done
```

**Check:** `$R/stl_{ebqu,cl,mem}/target/release/libstl_bridge.so` exist.

### 7.2 Data

```bash
bash scripts/download_ext.sh                                                   # GNILC 353 GHz and the BK14 mask, 1.6 GB
sbatch $CPU_OPTS -o $RUNS/slurm.%j.log scripts/prepare_data_hires.slurm       # nside 512 and 1024, 55 GB, $R/data/maps_hi
python3 -m venv --system-site-packages $R/pysm_env && $R/pysm_env/bin/pip install pysm3==3.4.6
PYSM_PY=$R/pysm_env/bin/python sbatch -p <cpu> -c 8 --mem=96G -t 6:00:00 -o $RUNS/slurm.%j.log --wrap "bash scripts/reproduce.sh sky"
```

`reproduce.sh sky` writes the PySM d12 sky (`$R/sky_d12`, 2 min; the 5.7 GB of templates go to `~/.astropy/cache`
at first use) and the mock observation (`$R/sim_d12`: truth, d = truth + FFP10 simulation 0, links to the other files of `$R/data/maps`).
Put `HIRES=...` in the environment if the nside 512/1024 maps are not in `$R/data/maps_hi`.

**Check:** `$R/data/ext` holds the GNILC map and `bk14_mask_gal_n0512.fits`; `$R/data/maps_hi` holds 938 files;
the `sky` log prints `E: amplitude gain 1.663 at l=100, slope +0.082` and `B: amplitude gain 1.640 at l=100, slope +0.068`,
and after the correction `C(s)/C(d1xd2)` between 0.86 and 1.10 in every band.

### 7.3 Trainings

`bash scripts/reproduce.sh runs` submits them with the other runs (one A100 each):

| run | `STL` | time (A100) | GPU memory | host memory | check: last `eval 3000` line |
|---|---|---|---|---|---|
| `sim_d12_ebqu` | `stl_cl` | 2 h 27 | < 40 GB | 48 GB | `Q err/noise 0.862 U err/noise 0.752` |
| `n256_ebqu_c16` | `stl_ebqu` | 3 h 02 | < 40 GB | 48 GB | Q 0.81, U 0.75 |
| `n256_ebqu_c16cl` | `stl_cl` | 2 h 25 | < 40 GB | 48 GB | Q 0.80, U 0.78 |
| `n512_ebqu_c16` | `stl_mem` | 7 h 49 | 48.4 GB | 64 GB | `Q err/noise 0.860 U err/noise 0.846` |
| `n1024_ebqu_c16` | `stl_mem` | not yet measured | about 50 GB (estimate) | 160 GB | |

The value is the rms of d - u over that of the noise. A rerun differs within the spread between trainings
(about 0.02), since the GPU sums are not bit-reproducible.

### 7.4 Known sky: other noises

```bash
bash scripts/reproduce.sh apply     # on a GPU node, after sim_d12_ebqu
```

**Check:** `$RUNS/sim_d12_ebqu/eval_truth.txt` gives, for every `u_mc*.npz`, `u-s rms/noise by |b|` close to
`0.985 0.726 0.581 0.421 0.369`, and the EE and BB ratios of Table 3 of the paper.

### 7.5 Figures

```bash
bash scripts/reproduce.sh figures
```

The spectra go up to l = 3 nside - 1. Copy into `text/figures/`:

| written by `reproduce.sh figures` | paper |
|---|---|
| `$RUNS/figs_sim_d12/sim_{maps,spectra}.pdf` | `sim_{maps,spectra}.pdf` |
| `$RUNS/figs_ref/real_{bicep,actpol,maps,spectra}.pdf` | `ref_{bicep,actpol,maps,spectra}.pdf` |
| `$RUNS/figs_n512/real_{bicep,actpol,maps,spectra}.pdf` | `n512_{bicep,actpol,maps,spectra}.pdf` |

`make_figures_real.py` also prints, for each field, the correlation and rms of u and d against GNILC
(BICEP2 field at nside 256: `corr(u, G) 0.946` in Q, `0.933` in U).

## Pitfalls met while making the paper

- **Line endings.** A script edited on Windows and copied with CRLF line endings breaks bash (`$'\r': command not found`). Fix it with `sed -i 's/\r$//' scripts/*.sh scripts/*.slurm`.
- **Full work file system.** When the file system of `$R` is full, `torch.save` fails (`iostream error`), the run stops at its first checkpoint, and the Slurm log stays empty. Check with `dd if=/dev/zero of=$R/x bs=1M count=100 conv=fsync`. Then set `RUNS` to another file system (the outputs are about 40 MB per run). Since this happened, `model.pt` is written to a temporary file and renamed.
- **GPU memory.** The GPU disc engine reserves half of the free GPU memory for its geometry cache when it starts. On a GPU shared with the network, give `--cache-bytes` explicitly.
- **Home quota.** On Datarmor, `$HOME` has a quota of about 100 GB: a job that fills it fails at its first write with an empty log. The nside 512/1024 data, the known sky and the runs were written to `/work` (`RUNS=/work/...`, `HIRES=...`).
- **Long command lines.** Pasting a script through a shell that joins lines can replace `\` + newline by spaces; the command still runs, but check `bash -n scripts/reproduce.sh` after editing.
- **Mock sky.** In Sects. 3-4 the "clean sky" `s` of the evaluation is the PR3 map itself, which contains the Planck noise. Pixel errors and maps there compare losses, not the quality of a denoised dust map. The PR3 maps denoised (`--real`) are the actual test.
