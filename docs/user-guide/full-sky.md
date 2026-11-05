# Full-sky training

`stlia-train-denoise` (or `python train_denoise.py`) trains the spherical U-Net on whole maps. Both the network and the statistics run on one GPU. All options are listed in {doc}`../cli`.

## Mock data

The sky is a PR3 map of `--train-field`. Every step draws an observation $d = s + n_0$ and evaluates $u = f(d)$. It then draws `--realizations` noise maps and takes one Adam step on the weighted loss terms.

```bash
# nside 64, uniform white noise, auto loss: about 10 min on an A100 or a T4
stlia-train-denoise --data maps --out runs/n64 --nside 64 --noise uniform --loss data=1

# nside 256, FFP10 draws, the three cross losses of Delouis et al. (2022): about 2.3 h on an A100
stlia-train-denoise --data maps --out runs/n256 --nside 256 --noise pr3sims --loss l1=1,l2=1,l3=1
```

The noise models (`--noise`) are:

| model | noise |
|---|---|
| `uniform` | white, $\sigma$ = `--noise-rel` × std of the training sky, in every pixel |
| `pr3white` | white, with the per-pixel std of the PR3 simulations |
| `pr3sims` | the FFP10 simulations themselves, with a random sign: training draws from `--sims-train`, evaluation from `--sims-test` |

`--noise-scale` multiplies every model.

The network is evaluated every `--eval-every` steps, on `--train-field` and on `--test-field`, a sky it never sees in training. It is compared with the clean sky (see {doc}`outputs`).

## Real data

With `--real`, the observations are the PR3 maps themselves:

- the full mission is $d$;
- the half missions are $d_1$ and $d_2$;
- the network is trained on both fields, alternately.

This is the denoising of the actual data, as in Delouis et al. (2022):

```bash
stlia-train-denoise --data maps --out runs/real256 --nside 256 --real --loss data=1,l1=1,l2=1,l3=1
```

Keep the `data` term: it constrains the mean and the variance, which the cross statistics do not. The evaluation reports $\mathrm{rms}(d-u)$ over the noise rms, and the statistics of $u$ against those of $d$. `scripts/make_figures_real.py` then draws:

- the denoised maps;
- the BICEP2 field;
- the EE and BB spectra over $|b| > 30^\circ$. On these spectra, the half-mission cross spectrum $d_1 \times d_2$, free of noise bias, is the reference for $u$.

## The L-BFGS reference

The same command, with `--optimizer lbfgs`, minimises the same losses over the pixels of $u$ without a network, using the native L-BFGS of STL-Dev-MPI (see {doc}`../method`). Run it with the options of the network run it is compared with:

```bash
stlia-train-denoise --data maps --out runs/lbfgs256 --nside 256 --real --loss data=1,l1=1,l2=1,l3=1 \
    --realizations 4 --optimizer lbfgs --lbfgs-rounds 10 --lbfgs-iter 50
```

It writes the same files as a training run, except `model.pt`: `log`, `train.json` (one evaluation per round) and `eval_{Q,U}.npz` (with `u` the optimised maps), so `scripts/make_figures_real.py` draws its figures too.

On a T4, at $N_{\rm side}=256$ with 2 realisations, one evaluation of the loss for both fields takes about 23 s; 8 iterations bring the loss from 13.9 to 2.6.

## Frozen biases, spectrum loss and conditioning

The options below combine with the real-data mode (see {doc}`../method`):

```bash
# frozen noise biases (Delouis et al. 2022), the binned half-mission cross spectrum
# under three galactic masks, and the 857 GHz intensity as a conditioning map
stlia-train-denoise --data maps --out runs/real256_fz --nside 256 --real     --loss data=1,l1=1,l2=1,l3=1,cl=1 --bias frozen --bias-realizations 40 --bias-every 50     --cl-masks 0,30,45 --cond log857

# the L-BFGS reference with the same frozen biases and spectrum loss
stlia-train-denoise --data maps --out runs/lbfgs256_fz --nside 256 --real     --loss data=1,l1=1,l2=1,l3=1,cl=1 --bias frozen --optimizer lbfgs
```

## A joint (Q, U) network

With `--joint`, one network denoises Q and U together (`UNet(maps=2)`): both maps are its input and its output, and every step trains both. The scattering terms are those of each field, summed. With `--cl-spin 2`, the spectrum loss is on the EE and BB cross spectra of the half missions instead of the spectra of Q and U:

```bash
stlia-train-denoise --data maps --out runs/real256_joint --nside 256 --real --joint \
    --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-spin 2 --cl-backend stl --cl-masks 0,30,45 --cond log857
```

The EE/BB spectra need the `SpectrumEngine` of STL-Dev-MPI with spin spectra (commit 92f2aef or later). They run on CPU threads (`STL_THREADS`).

With `--pol-stats eb`, the terms are computed on the E and B maps of the output instead of Q and U, which makes them independent of the frame of Q and U (the spin-2 statistics of FOSCAT); the spectrum term (`--cl-spin 0`) is then on EE and BB:

```bash
stlia-train-denoise --data maps --out runs/real256_eb --nside 256 --real --joint --pol-stats eb \
    --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45,60 --cond log857+sigma \
    --bias frozen --bias-warmup 500 --stat-masks 0,20,30,45
```

`--cond sigma` adds the local noise level as a conditioning map, and `--bias-warmup 500` trains the first 500 steps with live biases before the first frozen estimate. `--init RUN/model.pt` starts from the weights of a previous run with the same network options.
Add `--stat-masks 0,20,30,45` to average every scattering statistic under the four galactic masks as well. This needs STL-Dev-MPI 9038114 or later; see {doc}`../installation`.

`--cond log857` reads `planck857_I_n<nside>.npy`, which is already needed by `l3`. With `--cl-backend torch`, the spectrum loss keeps its Legendre tables on the GPU: about 2 GB at $N_{\rm side}=256$, $\ell_{\max}=511$, in double precision. The `stl` backend stores no table.

## Resources

Measured times per step:

| $N_{\rm side}$ | loss | time per step |
|---|---|---|
| 64 | `data` | 0.2 s (A100 or T4) |
| 256 | `data` | 1.0 s (A100) |
| 256 | `l1,l2,l3` | 2.8 s (A100) |
| 256 | `data,l1,l2,l3` (`--real`) | 3.3 s (A100) |

The $N_{\rm side}=64$ runs fit on a 16 GB T4. The $N_{\rm side}=256$ runs of the paper used an 80 GB A100; we have not measured their peak memory.
