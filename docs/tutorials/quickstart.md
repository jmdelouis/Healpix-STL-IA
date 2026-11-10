# Quickstart: denoise the PR3 maps at nside 64

This tutorial trains the reference network of the paper on the Planck PR3 353 GHz $Q$ and $U$ maps degraded to
$\nside=64$, on one GPU. It needs about 25 GB of disk space for the downloads. Building the engine takes 10 min, the
downloads depend on the network, and the training takes 15 min on a T4. Every later tutorial reuses this setup.

## 1. Work directory and environment

```bash
export R=$HOME/stlia                     # any directory with space
mkdir -p $R && git clone https://github.com/jmdelouis/Healpix-STL-IA $R/src
export PY=python3                        # python >= 3.10 with numpy, scipy, healpy, matplotlib (torch optional)
export CUDARC_CUDA_VERSION=12080         # CUDA version of the driver (nvidia-smi: 12.8 -> 12080)
source $R/src/scripts/env.sh             # RUNS=$R/runs, STL=stl, G=$R/stl, PYTHONPATH, STL_NATIVE_LIBRARY
cd $R/src
```

On a cluster, put the module loads in a site file and give it as `SITE_ENV` (see `scripts/hpc_env.sh`).

## 2. The engine

```bash
git clone https://github.com/jmdelouis/STL-Dev-MPI $G/src
git -C $G/src checkout 2026.11.01
bash scripts/build_bridge.slurm          # or: sbatch $CPU_OPTS scripts/build_bridge.slurm
python3 -c "from stl_port.native import NativeUNet; print('ok')"
```

At run time the GPU kernels are compiled with NVRTC: `libnvrtc` of the driver's CUDA version must be on
`LD_LIBRARY_PATH` (a CUDA module, or the PyPI wheel `nvidia-cuda-nvrtc-cu12`).

## 3. Data at nside 64

Download the maps and the first 16 noise simulations of each 353 GHz map. Simulations 0-9 are kept for the evaluation,
and 10-15 are used for training.

```bash
bash scripts/download_planck.sh 16
P=$R/data/planck; M=$R/data/maps
$PY scripts/prepare_planck.py $P/HFI_SkyMap_353-psb_2048_R3.01_full.fits $M --name planck353 --nside 64
for h in 1 2; do
  $PY scripts/prepare_planck.py $P/HFI_SkyMap_353-psb_2048_R3.01_halfmission-$h.fits $M \
      --name planck353hm$h --fill-from planck353 --nside 64
done
$PY scripts/prepare_planck.py $P/HFI_SkyMap_857_2048_R3.01_full.fits $M --name planck857 --fields I --scale 1 --nside 64
$PY scripts/noise_std.py $M $P/noise/ffp10_noise_353_psb_full_map_mc_000??.fits --nside 64
for h in 1 2; do
  $PY scripts/noise_std.py $M $P/noise_hm/ffp10_noise_353_psb_hm${h}_map_mc_000??.fits --name noisehm$h --nside 64
done
ls $M | wc -l
```

`scripts/prepare_data.slurm` does the same for 50 simulations at $\nside=256$, 128 and 64.

## 4. Train

The options are those of the reference at $\nside=256$ ({doc}`pr3-reference`), scaled to $\nside=64$. Four levels
are used, and the spectrum term goes up to $3\nside-1=191$.

```bash
$PY $G/src/python/stl_ensemble/train_native_unet.py --data $M --out $RUNS/q64 \
  --nside 64 --pol ebqu --channels 16,32,48,64 --cond log857+sigma \
  --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45 --cl-masks 0,30,45,60 --cl-lmax 191 \
  --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-15 \
  --optimizer adam --steps 1000 --lr 1e-3 --seed 0 \
  --input-scale map --noise-pools host --engine-cache-bytes 0 --eval-every 100 --log-every 50 \
  | tee $RUNS/q64.log
```

On a T4 it prints, among other lines:

```text
nside 64 J 5 levels 4 channels [16, 32, 48, 64] params 229666 sims 6 pol ebqu loss {... 'cl': 1.0} optimizer adam ...
step 50 joint loss 3.28697 data 0.437 l1 0.6477 l2 0.2224 l3 0.3089 cl 1.671 1.695 s/step
...
step 1000 joint loss 0.342207 data 0.0479 l1 0.2545 l2 0.01852 l3 0.02068 cl 0.000604 0.923 s/step
eval 1000 t=923.0s E err/noise 0.567 B err/noise 0.519 Q err/noise 0.535 U err/noise 0.559 joint loss 0.34221
```

How to read it:
- The first 200 steps use live noise biases and take about 1.7 s each. The frozen-bias steps then take 0.7 s.
- `err/noise` is the rms of the removed part $d-u$ over that of the noise. At $\nside=64$ the sky dominates, so the
  network removes about half the noise rms. At $\nside=256$ it removes 0.87 ($Q$) and 0.79 ($U$).
- The GPU sums are not bit-reproducible, so a rerun gives values within about 0.02.

## 5. Look at the result

```bash
$PY scripts/make_figures_real.py $RUNS/q64 $M $RUNS/figs_q64 --nside 64 --sims 0-9
$PY scripts/text_numbers.py $RUNS/q64 $M 64
```

The first command writes `real_maps.pdf`, `real_bicep.pdf`, `real_actpol.pdf` and `real_spectra.pdf`. A good run has
two properties:
- $d-u$ looks like the half-mission difference $(d_1-d_2)/2$, an independent estimate of the noise;
- the spectra of $u$ follow the half-mission cross spectrum $d_1\times d_2$, which has no noise bias.

`text_numbers.py` prints these comparisons as numbers (see {doc}`../options/scripts`). Its multipole bins are those of $\nside=256$: the bins above $\ell=191$ are empty at $\nside=64$ and print `nan`.

## Next

- Change an option and rerun: every option is in {doc}`../options/native-trainer`, with what it changed in the paper.
- {doc}`pr3-reference`: the same at $\nside=256$, with 40 simulations.
