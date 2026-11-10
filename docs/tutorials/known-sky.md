# Validation on a known sky

On the PR3 maps the sky is unknown. This tutorial builds a mock observation whose sky is known:
- **sky:** the PySM 3 dust model `d12`, given the EE and BB spectra of the PR3 data;
- **noise:** the FFP10 noise simulation 0 is added to it;
- **training:** the network is trained on this observation;
- **test:** the trained network is applied to the same sky with the noise simulations 1-9, never seen in training, and
  compared with the truth.

This is paper Sect. 4.

## 1. The sky and the observation

PySM needs its own environment (`pip install pysm3==3.4.6`). It downloads 5.7 GB of templates at first use, and the
sky needs 96 GB of memory.

```bash
python3 -m venv --system-site-packages $R/pysm_env && $R/pysm_env/bin/pip install pysm3==3.4.6
PYSM_PY=$R/pysm_env/bin/python bash scripts/reproduce.sh sky
```

`reproduce.sh sky` runs two steps:
- `make_pysm_sky.py $R/sky_d12 --model d12 --nside 256 512 1024`;
- `make_sim_obs.py $R/sky_d12 $R/data/maps $R/sim_d12 --nside 256 --mc 0`.

The log prints the gains applied to E and B, `1.663` and `1.640` at $\ell=100$. `$R/sim_d12` holds the observation
under the PR3 file names, and the truth `truth353_{Q,U}_n256.npy`.

## 2. Train on the observation

The options are exactly those of the reference ({doc}`pr3-reference`), with `--data $R/sim_d12`:

```bash
sbatch $GPU_OPTS -J sim_d12_ebqu_cl3n -o $RUNS/slurm.%j.log scripts/train_native.slurm sim_d12_ebqu_cl3n \
  --nside 256 --pol ebqu --channels 16,32,48,64,64,64 --cond log857+sigma \
  --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45 --cl-masks 0,30,45,60 --cl-lmax 767 \
  --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-49 \
  --optimizer adam --steps 3000 --lr 1e-3 --seed 0 \
  --input-scale map --noise-pools host --engine-cache-bytes 0 --mem-trace --eval-every 500 --log-every 50 \
  --data $R/sim_d12
```

Check (2 h 10 on an A100): `eval 3000 ... Q err/noise 0.902 U err/noise 0.834`.

## 3. Apply to unseen noises and compare with the truth

```bash
$PY scripts/apply_native.py $RUNS/sim_d12_ebqu_cl3n $R/sim_d12 $RUNS/sim_d12_ebqu_cl3n/apply/u.npz --mcs 1-9
$PY scripts/eval_truth.py $R/sim_d12 $R/sim_d12 $RUNS/sim_d12_ebqu_cl3n \
    --maps $RUNS/sim_d12_ebqu_cl3n/apply/u_mc*.npz | tee $RUNS/sim_d12_ebqu_cl3n/eval_truth.txt
$PY scripts/mean_truth.py $RUNS/sim_d12_ebqu_cl3n/eval_truth.txt
$PY scripts/make_figures_sim.py $R/sim_d12 $RUNS/sim_d12_ebqu_cl3n $RUNS/figs_sim_d12 \
    --apply "$RUNS/sim_d12_ebqu_cl3n/apply/u_mc*.npz"
```

`apply_native.py` must be given the network options of the training. The defaults are those of the reference;
otherwise pass `--channels`, `--cond`, `--pol`.

**What to expect** (paper Table 2, means over mc1-9):
- **Pixel error:** the rms of $u-s$ over that of the noise is 0.98, 0.68, 0.55, 0.39 and 0.32 in the latitude bands
  $\vert b\vert<20^\circ$, 20-30, 30-45, 45-60 and $>60^\circ$. It is the same on every noise, to 0.006.
- **Spectra:** over $\vert b\vert>30^\circ$, the spectra of the truth are recovered to 4 % (EE) and 2 % (BB) up to
  $\ell=300$, and to 10 % up to 600.
- **Small scales:** where the noise dominates, $u$ has the right spectrum but is only partly correlated with the truth
  (0.54/0.43 for $300\le\ell<450$).

This test measures what the loss recovers. A change of an option can be judged here against the truth, before running
it on the PR3 maps.
