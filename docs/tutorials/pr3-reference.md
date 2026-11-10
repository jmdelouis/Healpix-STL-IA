# The reference at nside 256

The reference network of the paper, `n256_ebqu_c16_cl3n`: the PR3 353 GHz $Q$ and $U$ maps at $\nside=256$, with
40 noise simulations for training and 10 for evaluation. It takes 2 h 11 on an A100 (under 40 GB of GPU memory).
Start from the {doc}`quickstart` setup (environment and engine).

## Data

```bash
bash scripts/download_planck.sh 50                          # 60 GB in $R/data/planck
sbatch $CPU_OPTS -o $RUNS/slurm.%j.log scripts/prepare_data.slurm   # 3 h on 4 cores; or bash scripts/prepare_data.slurm
bash scripts/download_ext.sh                                # GNILC and the BK14 mask, for the figures
```

## Train

```bash
sbatch $GPU_OPTS -J n256_ebqu_c16_cl3n -o $RUNS/slurm.%j.log scripts/train_native.slurm n256_ebqu_c16_cl3n \
  --nside 256 --pol ebqu --channels 16,32,48,64,64,64 --cond log857+sigma \
  --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45 --cl-masks 0,30,45,60 --cl-lmax 767 \
  --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-49 \
  --optimizer adam --steps 3000 --lr 1e-3 --seed 0 \
  --input-scale map --noise-pools host --engine-cache-bytes 0 --mem-trace --eval-every 500 --log-every 50
```

Without Slurm, run `bash scripts/train_native.slurm n256_ebqu_c16_cl3n ...` with the same options. The log is
`$RUNS/n256_ebqu_c16_cl3n/log`. Its last evaluation line is the check of the run:

```text
eval 3000 t=7864.3s E err/noise 0.837 B err/noise 0.848 Q err/noise 0.866 U err/noise 0.789 joint loss 3.8909
```

A rerun gives values within about 0.02 (the GPU sums are not bit-reproducible). Each choice of this command, and what
changed when it was varied, is in {doc}`../options/native-trainer`.

## Figures and numbers

```bash
E=$R/data/ext
$PY scripts/prepare_gnilc.py $E/COM_CompMap_IQU-thermaldust-gnilc-varres_2048_R3.00.fits $E
$PY scripts/make_figures_real.py $RUNS/n256_ebqu_c16_cl3n $R/data/maps $RUNS/figs_ref \
    --gnilc $E --field-mask $E/bk14_mask_gal_n0512.fits
$PY scripts/make_figures_real.py $RUNS/n256_ebqu_c16_cl3n $R/data/maps /tmp/f60 --bcut 60   # spectra over |b| > 60 deg
$PY scripts/text_numbers.py $RUNS/n256_ebqu_c16_cl3n $R/data/maps 256
```

What the paper reports for this run (Sect. 5, Table 3), to compare with:

| measure | value |
|---|---|
| removed rms over noise rms, $Q$ / $U$ | 0.87 / 0.79 |
| EE/BB of $u$ over the cross spectrum, $\vert b\vert>30^\circ$ | within 2 % up to $\ell=300$; 0.94/0.95 for $300\le\ell<450$ |
| same, $\vert b\vert>60^\circ$, $300\le\ell<450$ | 0.85/1.56 (the BB excess of the faint sky) |
| EE of $d-u$ over the noise, $30\le\ell<60$, $\vert b\vert>30^\circ$ | 0.72 |
| correlation of $u$ with GNILC, BICEP2 field, $Q$ / $U$ | 0.946 / 0.943 (0.674 / 0.601 for $d$) |
| correlation of $u$ with GNILC, ACTPol D56 field | 0.911 / 0.820 |

How to read them:
- **Removed part:** $d-u$ should have the level and the spectrum of the noise. A ratio $(d-u)/n$ well above 1 at
  large scales means that sky was removed.
- **Signal kept:** the spectrum of $u$ should follow $d_1\times d_2$, the half-mission cross spectrum, which has no
  noise bias.
- **GNILC:** it is smoothed to about 60', so its correlation with $u$ is a consistency check, not a truth.
