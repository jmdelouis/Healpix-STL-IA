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

## Resources

Measured times per step:

| $N_{\rm side}$ | loss | time per step |
|---|---|---|
| 64 | `data` | 0.2 s (A100 or T4) |
| 256 | `data` | 1.0 s (A100) |
| 256 | `l1,l2,l3` | 2.8 s (A100) |
| 256 | `data,l1,l2,l3` (`--real`) | 3.3 s (A100) |

The $N_{\rm side}=64$ runs fit on a 16 GB T4. The $N_{\rm side}=256$ runs of the paper used an 80 GB A100; we have not measured their peak memory.
