# nside 512 and 1024

At $\nside=512$ the reference uses 48 GB of GPU memory and takes 8 h on an A100. At 1024 it fits on an 80 GB A100,
but it has not been trained yet. This page gives the extra options these resolutions need, and the problems seen at
512 (paper Sect. 5, "Resolution $\nside=512$", and Sect. 6).

## Data

```bash
sbatch $CPU_OPTS -o $RUNS/slurm.%j.log scripts/prepare_data_hires.slurm   # nside 512 and 1024, 55 GB, $R/data/maps_hi
export HIRES=$R/data/maps_hi
E=$R/data/ext; $PY scripts/prepare_gnilc.py $E/COM_CompMap_IQU-thermaldust-gnilc-varres_2048_R3.00.fits $E --nside 512
```

## The reference at nside 512

```bash
sbatch -p gpu --gres=gpu:a100:1 -c 6 --mem=64G -t 36:00:00 -J n512_ebqu_c16_cl3n_g -o $RUNS/slurm.%j.log \
  scripts/train_native.slurm n512_ebqu_c16_cl3n_g \
  --nside 512 --pol ebqu --channels 16,32,48,64,64,64,64 --cond log857+sigma \
  --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45 --cl-masks 0,30,45,60 --cl-lmax 1535 \
  --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-49 \
  --optimizer adam --steps 3000 --lr 1e-3 --seed 0 --spike-guard 3 \
  --input-scale map --noise-pools host --engine-cache-bytes 0 --mem-trace --eval-every 500 --log-every 50 \
  --data $HIRES
```

Compared with $\nside=256$:
- **one more level:** 7 channel entries;
- **`--cl-lmax 1535`** ($3\nside-1$);
- **`--noise-pools host`:** the 40 simulations no longer fit on the GPU;
- **`--spike-guard 3`.**

**Check:** `eval 3000 ... Q err/noise 0.976 U err/noise 0.888`, in 7 h 54 of training. The log also has lines
`spike guard: step ...`; the run of the paper had them at steps 247, 249, 1044 and 1046.

### Why the spike guard

The first training with these options had a loss spike at step 1200, from 11.5 to 46. Adam never recovered from it,
and the run ended at a loss of 11.2. The removed part was then 3.3 (EE) and 3.4 (BB) times the noise spectrum for
$10\le\ell<30$: sky was removed at large scales.

With `--spike-guard 3`, a frozen-bias step whose loss is more than 3 times the running loss triggers three actions:
- the weights go back to the last `--log-every` mark;
- the Adam moments are reset;
- the learning rate is halved.

The run then ends at 8.99, with a large-scale residual of 1.5 (EE) and 2.4 (BB). The cause of the spikes was not
studied.

### What is left at 512

These problems remain (paper Sect. 5, `HANDOFF.md`):
- **BB too high:** BB of $u$ is 1.3-1.6 times the cross spectrum for $450\le\ell<1024$ over $\vert b\vert>30^\circ$,
  against 1.15-1.22 at 256.
- **Rise above $\ell=2\nside$:** $u$ rises above the cross spectrum there (2.5 in EE and 7 in BB for
  $1024\le\ell<1280$). The cross spectrum there is a few per cent of the noise, and the removed part is 0.98-0.99 of
  the noise spectrum.
- **Lower GNILC correlation:** at full resolution, the correlation with GNILC is lower than at 256 (BICEP2 field 0.88
  and 0.87), since $u$ keeps the scales GNILC smooths out. Averaged to $\nside=256$ (mean of the 4 NESTED children), it
  is within 0.02 of the $\nside=256$ reference.

## Memory

| $\nside$ | GPU memory (peak) | host memory | step, live / frozen biases | options |
|---|---|---|---|---|
| 256 | under 40 GB | 48 GB requested | 2.6 s average | -- |
| 512 | 48.4 GB | 19 GB at setup, `--mem=64G` | 19 s / 8.0 s | `--noise-pools host` |
| 1024 | 50.5 GB at the first live step | `--mem=160G` | 174 s (first live step) | `--noise-pools host --live-batch 4 --bias-batch 2` |

`--mem-trace` prints the peak of every phase. If a run is short of memory:
- lower `--bias-batch` (draws per statistics call of a bias estimate);
- lower `--live-batch` (pairs per call of the live cross terms).

Both cost more calls, not more memory.

## nside 1024 (not trained yet)

```bash
sbatch -p gpu --gres=gpu:a100:1 -c 6 --mem=160G -t 96:00:00 -J n1024 -o $RUNS/slurm.%j.log \
  scripts/train_native.slurm n1024_ebqu_c16_cl3n_g \
  --nside 1024 --pol ebqu --channels 16,32,48,64,64,64,64,64 --cond log857+sigma \
  --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45 --cl-masks 0,30,45,60 --cl-lmax 3071 \
  --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-49 \
  --optimizer adam --steps 3000 --lr 1e-3 --seed 0 --spike-guard 3 \
  --input-scale map --noise-pools host --live-batch 4 --bias-batch 2 --engine-cache-bytes 0 --mem-trace \
  --eval-every 500 --log-every 50 --data $HIRES
```

The first live step took 174 s, so the 200 live steps alone take about 10 h. Two ways to shorten it were not tested:
fewer live steps (`--bias-warmup`), or fewer steps in all (`--steps`). See "What to try first" in {doc}`../options/index`.
