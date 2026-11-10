# `train_native_unet.py`: the production trainer

`train_native_unet.py` is the driver of the native U-Net of STL-Dev-MPI (`stl_port.native.NativeUNet`). It trains the
network, the statistics, their gradient and the optimiser on one GPU. Every map of the paper comes from it. The file
lives in STL-Dev-MPI, `python/stl_ensemble/train_native_unet.py`, at the tag **`2026.11.01`**. This page documents that
version.

```bash
python $G/src/python/stl_ensemble/train_native_unet.py --data DATA_DIR --out RUN_DIR [options]
# or, on a Slurm cluster, from this repository:
STL=stl sbatch $GPU_OPTS -o $RUNS/slurm.%j.log scripts/train_native.slurm RUN [options]
```

For each option the tables below give:
- the **default**;
- the value of the **reference** run `n256_ebqu_c16_cl3n` of the paper;
- **what was measured** when it was changed, with the section of the paper that reports it.

The paper sections cited are: 2 Method, 3 Data and reference training, 4 Validation on a known sky, 5 Results on the
PR3 maps, 6 Native implementation and cost, 7 History of the tests and of the choices (with Table 5, the choices).
Numbers are from the PR3 maps at $\nside=256$ unless stated otherwise:
- $\sigma$: rms of the removed part $d-u$ over that of the noise;
- $u/x$: spectrum of $u$ over the half-mission cross spectrum;
- $(d-u)/n$: spectrum of the removed part over the noise spectrum.

The reference command is:

```bash
train_native_unet.py --data $R/data/maps --out $RUNS/n256_ebqu_c16_cl3n \
  --nside 256 --pol ebqu --channels 16,32,48,64,64,64 --cond log857+sigma \
  --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45 \
  --cl-masks 0,30,45,60 --cl-lmax 767 \
  --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-49 \
  --optimizer adam --steps 3000 --lr 1e-3 --seed 0 \
  --input-scale map --noise-pools host --engine-cache-bytes 0 --mem-trace \
  --eval-every 500 --log-every 50
```

Several defaults differ from the reference (`--pol`, `--cl-masks`, `--cl-lmax`, `--bias-warmup`, `--cond`): always give
the options explicitly, as above.

## Data and run

| option | default | reference | meaning |
|---|---|---|---|
| `--data DIR` | required | `$R/data/maps` | directory of the input maps (see {ref}`data-files` below) |
| `--out DIR` | required | `$RUNS/<run>` | output directory (see {ref}`native-outputs`) |
| `--nside N` | 64 | 256 | resolution; the files `*_n{N}.npy` are read |
| `--sims A-B` | `10-49` | `10-49` | FFP10 simulations used for the noise draws (biases, live steps, conditioning); only those present for the full mission and both half missions are used. Keep 0-9 out: they are the evaluation simulations. |
| `--seed S` | 0 | 0 | initial weights and draws. Two seeds end on maps that differ by a large fraction of the noise, with spectra within 0.02 over $\vert b\vert>30^\circ$ (paper Sect. 5, "The spread between trainings"). |
| `--real` | off | -- | accepted for compatibility with `stlia-train-denoise`; the PR3 maps (or a mock observation under the same names) are always the data |
| `--seconds T` | none | -- | stop the training after T seconds of training time (evaluations not counted) |

(data-files)=
### Files read from `--data`

All maps are NESTED `.npy`, in $\mu$K$_{\rm CMB}$ (857 GHz in MJy/sr), made by `prepare_data.slurm` or
`prepare_data_hires.slurm` (see {doc}`scripts`):

- `planck353_{Q,U}_n{N}.npy`: the observation $d$;
- `planck353hm1_{Q,U}_n{N}.npy`, `planck353hm2_...`: the half missions $d_1$, $d_2$ ($L_1$, $L_{\rm cl}$);
- `planck857_I_n{N}.npy`: the dust template $T$ ($L_3$, `--cond log857`);
- `planck353_noise_{Q,U}_n{N}_mc{k}.npy`, `planck353_noisehm1_...`, `planck353_noisehm2_...`: the FFP10 noise of
  simulation `k`, full mission and half missions.

A mock observation of a known sky (`make_sim_obs.py`) has the same names, so the trainer runs on it unchanged
({doc}`../tutorials/known-sky`).

## Network

| option | default | reference | meaning, and what was measured |
|---|---|---|---|
| `--channels LIST` | `32,64,96,128,...` | `16,32,48,64,64,64` | channels per level, finest first. 32-128 and 64-256 give the same spectra; the narrow network keeps the large scales better: $(d-u)/n=0.90$ against 1.54 for $10\le\ell<30$, with $6\times10^5$ parameters instead of $2.4\times10^6$ (Table 5). |
| `--levels L` | down to $\nside=8$ | 6 at 256, 7 at 512 | number of levels of the NESTED pyramid (one more per doubling of $\nside$) |
| `--ksz K` | 3 | 3 | kernel size of the stencil ($K\times K$ taps in the tangent plane); not varied |
| `--gauge` | `phi` | `phi` | orientation of the kernel: `phi` follows the local meridian, `cosmo` the other HEALPix convention. Both gauges are equivalent; planar convolutions on the base faces leave errors 1.16-1.36 times larger along the face edges (Table 5). |
| `--cond` | `none` | `log857+sigma` | maps added to the input and to every decoder level: `log857` the log of the 857 GHz map, `sigma` the log of the local noise rms over `--sims`. Null test: $\vert{\rm corr}(d-u,I_{857})\vert\le0.012$, no imprint of the template (Table 5, Sect. 5). |
| `--input-scale` | `map` | `map` | joint networks: divide each input map by its own std (`map`) or both by one std (`shared`); not compared |
| `--init MODEL` | none | -- | start from a `model.pt` trained at a lower (or the same) $\nside$: shared levels are copied, the extra coarse levels start fresh. At $\nside=128$ from a $\nside=64$ network, after 1000 steps the loss is 1.80 against 1.42-1.54 from scratch, at the same pixel error: no gain (`HANDOFF.md`). |
| `--init-cascade` | off | -- | with `--init`, the new coarse levels start as copies of the coarsest level of the small network; 2.17 after 1000 steps, no gain |

The network predicts a correction, $u=d-\sigma_d\,r_\theta(d/\sigma_d)$, and its last layer starts at zero, so the
untrained network is the identity (paper Sect. 2.2).

## Polarisation

| option | default | reference | meaning, and what was measured |
|---|---|---|---|
| `--pol` | `qu` | `ebqu` | `qu`: $Q$ and $U$ trained alternately, one network each, statistics on $Q$, $U$. `eb`: one network on the E and B maps. `ebqu`: one network $(Q,U)\to(Q,U)$ whose terms are computed on the harmonic E and B maps of its output (paper Sect. 2.3). `spin`: one network on $(Q,U)$, statistics of $\chi=\bar\eth^2(Q+iU)$. |
| `--pol-weight W` | 1 (0.5 for `spin`) | 1 | factor on the terms of each field |
| `--eb-iter I` | 3 | 3 | iterations of `healpy.map2alm` in the E/B transform |

What decided `ebqu` (Table 5 and items 7-14 of Sect. 7):
- **statistics on $Q$, $U$** (`qu`): B in excess at $\vert b\vert>60^\circ$, 2.65 times the cross spectrum for
  $300\le\ell<450$ against 1.38;
- **`spin`**: $\chi$ weights the multipoles as $\ell^4$ in power and does not see the large scales; the network
  removes sky there, $(d-u)/n=8$ for $10\le\ell<30$;
- **`eb`**: the part $P$ of a map lost in the E/B round trip (0.47 of the noise rms, at the pixel scale) passes
  through, so only 0.70 of the noise rms is removed, against 0.81 with `ebqu`.

## Loss terms

`--loss` takes comma-separated `term=weight` pairs. The terms are those of paper Sect. 2.1:

| term | compares | noise bias |
|---|---|---|
| `data` | $\Phi(d)$ with $\langle\Phi(u+n_i)\rangle$ (reduced $\chi^2$, ergodic weights) | on the variance and $S_1$-$S_4$ |
| `l1` | $\Phi(d_1,d_2)$ with $\langle\Phi(u+\tilde n_1,u+\tilde n_2)\rangle$ | not on $S_2$ |
| `l2` | $\Phi(d,u)$ with $\langle\Phi(u+\tilde n,u)\rangle$ | not on $S_2$ and one order of $S_3$ |
| `l3` | $\Phi(T,d)$ with $\langle\Phi(T,u+\tilde n)\rangle$, $T$ the 857 GHz map | not on $S_2$ and one order of $S_3$ |
| `cl` | binned $C_b(m\,d_1,m\,d_2)$ with $C_b(m\,u,m\,u)$ under galactic masks $m$ | none |

| option | default | reference | what was measured |
|---|---|---|---|
| `--loss` | `data=1,l1=1,l2=1,l3=1` | `data=1,l1=1,l2=1,l3=1,cl=1` | the cross terms alone let the mean of $u$ drift by 60 ergodic standard deviations; `data` fixes it. Without `cl`, $u$ reaches only 54-67 % of the EE cross spectrum for $200\le\ell<450$, for the network and for the per-map L-BFGS (Table 5). |
| `--j J` | $\log_2\nside-1$ | 7 | number of wavelet scales of the statistics |
| `--stat-masks LIST` | none (full sky) | `0,20,30,45` | every coefficient of every term is also averaged under $\vert b\vert>b_0$ for each $b_0$, with its own ergodic weight and bias. Without it, the full-sky weights are dominated by the Galactic plane and the faint sky is hardly constrained; 16-27 % more time (Table 5). |

Two terms were tried and rejected, and are not available here (Table 5):
- $C_b(u,d-u)=0$: it removes the large scales, $(d-u)/n=2.1$ for $30\le\ell<60$;
- $\Phi(d-u)$ against $\Phi(n)$ instead of `l2`: it breaks the data term (21 instead of 1.2) and doubles the BB excess
  at $\vert b\vert>60^\circ$.

## Spectrum term `cl`

| option | default | reference | meaning, and what was measured |
|---|---|---|---|
| `--cl-masks LIST` | `0,30,45` | `0,30,45,60` | galactic cuts of the apodised masks of the pseudo-spectra |
| `--cl-lmin L` | 10 | 10 | first multipole of the bins |
| `--cl-lmax L` | $2\nside-1$ | $3\nside-1$ (767 at 256, 1535 at 512) | last multipole. Up to $2\nside-1$, above $\ell=2\nside$ the spectra of $u$ rise to 1.54/2.23 (EE/BB) times the cross spectrum for $600\le\ell<768$, against 1.06/1.22 with $3\nside-1$, the same below $\ell=450$; the removed rms grows from 0.81/0.75 to 0.87/0.79 (Sect. 7, item 19). At $\nside=512$, u still rises above the cross spectrum for $\ell>1024$ (Sect. 5). |
| `--cl-width W` | 30 | 30 | bin width $\Delta\ell$ |
| `--cl-apod D` | 5 | 5 | FWHM (deg) of the mask apodisation |
| `--cl-engine` | `native` | `native` | `native`: spectra inside the trainer, on the GPU, one network pass per step. `python`: from the driver, one more forward and backward. Same values to $3\times10^{-15}$; 2.9 against 3.6 s per step (Sect. 7, item 15). |

With `--pol eb` or `ebqu` the spectra are those of the E and B maps (EE and BB constrained separately). On the spectra
of $Q$ and $U$, EE was 15 % low and BB 25-28 % high for $200\le\ell<450$ (Table 5).

## Noise biases

The model side of each scattering term needs the noise bias $\langle\Phi(u+n)\rangle-\Phi(u)$ (paper Sect. 2.4).
"Live" biases average the statistics of $u+n_i$ over a few fresh draws at every step, with the gradient through them.
"Frozen" biases are estimated with many draws at the current $u$, kept fixed, and re-estimated periodically.

| option | default | reference | meaning, and what was measured |
|---|---|---|---|
| `--bias-warmup N` | 0 | 200 | the first N steps use live biases, the frozen estimates start at step N+1. The first estimate at the untrained output, which still holds the noise, overestimates the bias: without `cl`, frozen biases from the start lose power at small scales (Sect. 7, item 6). A live step costs 2.6 times a frozen one (19 against 8 s at $\nside=512$). |
| `--realizations B` | 4 | 4 | live steps: noise draws per step |
| `--bias-realizations M` | 40 | 40 | draws of a frozen estimate |
| `--bias-every K` | 50 | 50 | steps between two frozen estimates (from the current output) |
| `--bias-all` | off | off | estimate a bias for every coefficient. By default the coefficients without noise bias (paper Table 1) get zero: their estimate would only add the scatter of the M draws. |
| `--bias` | `frozen` | `frozen` | only `frozen` (after the warm-up) in this trainer |
| `--bias-batch N` | 5 | 5 (2 at 1024) | draws per statistics call of a bias estimate; smaller uses less GPU memory |
| `--live-batch N` | 0 (all) | 0 (4 at 1024) | pairs per statistics call of the live cross terms; smaller uses less GPU memory, one more forward per call |

## Optimiser

| option | default | reference | meaning, and what was measured |
|---|---|---|---|
| `--optimizer` | `adam` | `adam` | `adam`: Adam with a cosine schedule. `lbfgs`: the `--bias-warmup` live steps with Adam, then L-BFGS rounds on the weights. `adam+lbfgs`: the whole Adam schedule, then L-BFGS rounds. At $\nside=64$, for the same 923 s, the loss (fixed set of 40 bias draws) ends at 0.479 with Adam, 0.534 with `lbfgs` and 0.488 with 500 Adam steps then L-BFGS: L-BFGS does not shorten the training (Sect. 7, item 18). |
| `--steps N` | 3000 | 3000 | Adam steps, also the length of the cosine schedule. 3000 rather than 1000: $\sigma=0.80$ against 0.69 on $Q$, same spectra (Table 5). At $\nside=64$ the loss is within 4 % of its final value at half the steps; at $\nside=512$ it was 9.54, 9.13 and 8.99 at steps 2000, 2500 and 3000. Fewer steps were not tested on the spectra. |
| `--lr LR` | $10^{-3}$ | $10^{-3}$ | initial learning rate; $lr_k=lr\,\tfrac12[1+\cos(\pi(k-1)/N)]$ |
| `--spike-guard F` | 0 (off) | 0 at 256, **3 at 512** | a frozen step whose loss exceeds F times the running loss (within one bias period) restores the parameters saved at the last `--log-every` mark, resets the Adam moments and halves the learning rate; each event is printed and saved in `history.json`. At $\nside=512$ without it, a spike at step 1200 (loss 11.5 to 46) was never recovered, leaving the removed part 3.3 times the noise for $10\le\ell<30$; with it, 1.5 (Sect. 5, "Resolution $\nside=512$"). |
| `--rounds R` | 20 | -- | L-BFGS rounds; biases refreshed at the start of each |
| `--lbfgs-iter I` | 20 | -- | iterations per round (about 22 loss-and-gradient evaluations) |
| `--lbfgs-history H` | 10 | -- | L-BFGS memory |
| `--lbfgs-max-eval E` | none | -- | evaluation cap per round |
| `--lbfgs-lr S` | 1 | -- | step scale of the strong-Wolfe line search |
| `--lbfgs-fields` | `alternate` | -- | `qu`: one field per round (`alternate`) or both (`joint`); the joint networks always train both. Alternating is the most erratic: each round degrades the other field. |

## Memory and device

| option | default | reference | meaning |
|---|---|---|---|
| `--backend` | `gpu` | `gpu` | `gpu` or `cpu` |
| `--device D` | 0 | 0 | GPU index |
| `--noise-pools` | `device` | `host` | keep the noise simulations on the GPU (uploaded once) or on the host (each statistics call uploads its draws). `host` is needed at $\nside\ge512$. |
| `--engine-cache-bytes B` | engine default | 0 | GPU geometry cache; 0 sizes it from the free memory |
| `--mem-trace` | off | on | peak GPU and host memory per phase (setup, bias estimate, live and frozen steps, evaluation), printed when a peak grows and saved in `train.json` |

Measured peaks of the reference options:
- 11.4 GB at $\nside=256$ for the network on the E and B maps (Sect. 6), under 40 GB for the reference;
- 48.4 GB at 512, with 19 GB of host memory at setup;
- 50.5 GB at 1024 at the first live step, with `--live-batch 4 --bias-batch 2` and 160 GB of host memory requested.

See {doc}`../tutorials/high-resolution`.

## Evaluation and logging

| option | default | reference | meaning |
|---|---|---|---|
| `--log-every N` | 10 | 50 | print a `step` line every N steps; also the snapshot period of `--spike-guard` |
| `--eval-every N` | 250 | 500 | evaluate every N steps (Adam) or rounds (L-BFGS) |
| `--eval-seconds T` | none | -- | also evaluate every T seconds of training |
| `--eval-draws N` | 0 | -- | also compute the loss with biases from one fixed set of N draws, the same for every run: the objective to compare runs or optimisers (used with 40 for the L-BFGS test) |

An `eval` line reads:

```text
eval 3000 t=7864.3s E err/noise 0.837 B err/noise 0.848 Q err/noise 0.866 U err/noise 0.789 joint loss 3.8909
```

`err/noise` is the rms of $d-u$ over that of the noise, for each map. On real data it measures how much is removed,
not an error, since the sky is unknown. `t` is the training time.

(native-outputs)=
## Outputs

| file | content |
|---|---|
| `log` (with `train_native.slurm`) | options, `step`, `eval`, `memory`, `spike guard` lines |
| `train.json` | options, number of parameters, noise rms, loss weights, times, loss history, evaluations, memory trace |
| `history.json` | every step (field, time, terms), evaluations, `spike_guard` events |
| `params.npy` | the weights, flat float32 (reload with `NativeUNet.set_params`, or `scripts/apply_native.py`) |
| `eval_{Q,U}.npz` | `d`, `u` (and `sky = d`): the observation and the denoised map; `eval_{E,B}.npz` with `--pol eb`, `ebqu` |
| `model.pt` | when torch is importable: the weights under the names of `stlia.unet.UNet`, with the network description; `stlia.unet.load` reads it |
