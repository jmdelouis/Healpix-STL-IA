# Method

## Statistical denoising

An observation is $d = s + n$: a sky $s$ plus a noise $n$ whose statistical model can be sampled. Let $\Phi$ be the scattering covariances of a map. They are complex coefficients, grouped in families:

- the mean and the variance;
- the first-order moduli $S_1$;
- the power-like $S_2$;
- the cross-scale and cross-orientation terms $S_3$ and $S_4$.

At $\nside=64$ with $J=5$ scales and 4 orientations there are 2522 of them, and at $\nside=256$ with $J=7$ there are 5882.

Following {ref}`Delouis et al. (2022) <delouis2022>`, a good estimate $u$ of $s$ is a map such that $u + n_i$ is *statistically indistinguishable* from $d$, for fresh noise realisations $n_i$. The auto loss (`data`) is the reduced $\chi^2$

$$
\mathcal{L}_{\rm data}(u; d) = \frac{1}{N_c}\sum_c
\frac{\bigl|\langle \Phi_c(u + n_i)\rangle_i - \Phi_c(d)\bigr|^2}{\sigma_c^2},
\qquad \sigma_c^2 = \frac{V_c(d)}{N_{{\rm pix},c}} ,
$$

where:

- the average runs over $B$ realisations (`--realizations`);
- $N_c$ is the number of coefficients;
- $\sigma_c$ is the *ergodic* standard deviation of coefficient $c$. It is the spatial variance $V_c(d)$ of the terms that coefficient averages, divided by the number of pixels of the pyramid level where it is computed. These are the weights of the ensemble synthesis of STL-Dev-MPI ({class}`stlia.stl_torch.DenoiseLoss`).

No pixel of $s$ enters the loss: the training data are noisy maps only.

## Amortisation

Instead of minimising $\mathcal{L}$ over the pixels of $u$ for one map, as Delouis et al. (2022) did with L-BFGS, we set $u = f_\theta(d)$. We then minimise the expectation of the loss over observations $d$ and noise draws, with respect to the network weights $\theta$, using Adam with a cosine schedule. Once trained, a map is denoised by one evaluation.

The network predicts a correction:

$$u = d - \sigma_d\, r_\theta(d/\sigma_d),$$

where $\sigma_d$ is the standard deviation of $d$, or a fixed scale for patch training. The last layer of $r_\theta$ starts at zero, so an untrained network is the identity.

## The per-map reference: L-BFGS without a network

`stlia-train-denoise --optimizer lbfgs` keeps the data, the losses and the evaluation of a training, but drops the network. It minimises the loss directly over the pixels of $u$, one map per field, starting from $u = d$. This is the per-map method of Delouis et al. (2022), and the reference the amortised network should approach.

- **Optimiser.** The native L-BFGS of STL-Dev-MPI (`stl_port.native.native_lbfgs`, strong-Wolfe line search), on $u/\mathrm{std}(d)$, which is of order one.
- **Objective.** The sum over the fixed observations of both fields of the weighted terms of `--loss`, exactly as in training.
- **Noise realisations.** They are drawn from the training simulations. Their seed is fixed within a round, so that the loss is deterministic for the line search, and changes between the `--lbfgs-rounds` rounds of `--lbfgs-iter` iterations. Unlike the original method, no noise bias is frozen: $\Phi(u+n_i)$ is recomputed at the current $u$ at every evaluation, and the gradient goes through it.
- **Evaluation.** Every round ends with the same evaluation as a training run, so the two can be compared directly in `train.json`.

## Noise biases: live or frozen

The model side of every term is an average over noise realisations, such as $\langle\Phi(u+n_i)\rangle$. Two schemes are available (`--bias`):

**`live`** (default). At every step, or every evaluation of the L-BFGS loss, the average uses $B$ = `--realizations` draws at the current $u$, and the gradient goes through it. The bias $\langle\Phi(u+n)\rangle-\Phi(u)$ is never formed explicitly: it follows $u$ at every step, estimated from a few draws.

**`frozen`**: the scheme of Delouis et al. (2022). At a fixed map $u_0$, the bias of each term,

$$B_k = \langle \Phi_k(u_0 + n_i) \rangle_{i \le M} - \Phi_k(u_0),$$

is estimated with $M$ = `--bias-realizations` draws (40 by default). The model is then $\Phi_k(u) + B_k$, with the gradient through $\Phi_k(u)$ only, and $B_k$ is kept constant until the next estimate:

- with L-BFGS, at the start of every round;
- for a network, every `--bias-every` steps, from the current output.

The loss needs no noise draw between two estimates, and each step is cheaper. The scheme needs fixed observations: `--real`, or `--optimizer lbfgs`.

## Masked statistics

With `--stat-masks 0,20,30,45`, every scattering coefficient of every term (`data`, `l1`, `l2`, `l3`) enters once per galactic mask $|b| > b_0$ ($b_0 = 0$: the full sky). Each entry has its own ergodic weight, computed from the variance and the number of pixels averaged under that mask (Delouis et al. 2022 used five masks).

- **What changes.** The wavelet fields are those of the full sky; only the averages that define the coefficients are restricted to the mask. At each coarser level, the mask contains the parents of the pixels of the finer one.
- **Why.** The full-sky averages are dominated by the Galactic plane; the masked ones constrain the faint high-latitude sky.
- **Engine.** The masked averages are computed by the STL-Dev-MPI engine (`NativeEngine(..., masks=...)`, from commit 9038114), with their gradient. They cost 16% (auto) to 27% (cross) more than the full-sky statistics for five masks.

## Spectrum loss

The term `cl` asks $u$ to have the power spectrum of the sky ({mod}`stlia.spectra`). Its noise-free target is the half-mission cross spectrum:

$$
C_b(m\,d_1, m\,d_2) \quad\text{vs}\quad \langle C_b(m(u+\tilde n_1), m(u+\tilde n_2))\rangle,
$$

for each galactic mask $m$ (`--cl-masks`, $|b|$ above each cut, apodised by `--cl-apod`). Here $C_b$ is the pseudo-spectrum of the masked maps, divided by the mean of the squared mask and binned in multipole (`--cl-lmin`, `--cl-lmax`, `--cl-width`).

Each bin is weighted by the Gaussian variance of the target, $(C_b^{11}C_b^{22} + (C_b^{12})^2)/\nu_b$, where $\nu_b$ is the number of modes of the bin times the effective sky fraction of the mask.

This term constrains the power at high Galactic latitude. The full-sky ergodic weights of the scattering statistics hardly do so, because the Galactic plane dominates their variance. With `--bias frozen`, the term is modelled as $C_b(u,u) + B_{\rm cl}$.

The spectra come from one of two differentiable `anafast` (`--cl-backend`), which agree with each other and with `healpy.anafast` to about $10^{-12}$:

- **`stl`**: the `SpectrumEngine` of STL-Dev-MPI (from commit 5648ba1), on the CPU or the GPU, with its exact VJP. `auto`, the default, uses it when the installed engine has it.
- **`torch`**: a ring-FFT spherical harmonic transform ({class}`stlia.sht.HEALPixSHT`), adapted from [healpix-analyse](https://github.com/GRID4EARTH/healpix-analyse) and FOSCAT, with gradients through `torch.autograd`.

On a T4 at $
side=256$, the spectra and gradient of 4 pairs of maps under 3 masks take 1.0 s with the engine, against 16 s with `torch`. They are the spectra of the scalar maps, Q and U separately, as the network treats them. E/B spectra (`spin=2`) are available in {class}`~stlia.sht.HEALPixSHT`, but a loss on them needs a joint (Q, U) network.

## Conditioning on the dust intensity

With `--cond log857`, the network also receives $\log I_{857}$, standardised: the 857 GHz intensity, a tracer of the dust and hence of the local signal-to-noise ratio. This map is concatenated:

- to the input of the network;
- averaged down the pyramid, to the input of every decoder level.

The decoding, i.e. how much is removed, can then depend on the local dust intensity. The untrained network is still the identity.

The 857 GHz map is also the template of `l3`. Check that the network does not copy its morphology into $u$: the residual $d-u$ must remain uncorrelated with it.

## The losses of Delouis et al. (2022)

`stlia-train-denoise --loss` combines weighted terms. Besides `data`, it implements the three cross-statistics terms of Delouis et al. (2022). Each is a reduced $\chi^2$ of the same form, with $\Phi(a, b)$ the cross scattering covariances of the engine and $\tilde n$ fresh noise draws:

| term | target | model | constrains |
|---|---|---|---|
| `l1` | $\Phi(d_1, d_2)$ | $\langle\Phi(u+\tilde n_1, u+\tilde n_2)\rangle$ | the statistics of $u$, from the two half missions $d_1, d_2$ |
| `l2` | $\Phi(d, u)$ | $\langle\Phi(u+\tilde n, u)\rangle$ | the alignment of $u$ with the data |
| `l3` | $\Phi(T, d)$ | $\langle\Phi(T, u+\tilde n)\rangle$ | the dependence on a template $T$ (the 857 GHz map) |

The noise biases of the original method are not frozen between batches of iterations: they are re-estimated at every step from the realisations.

The cross statistics of the engine have no mean or variance coefficient. With `l1`, `l2` and `l3` alone, the mean of $u$ is not constrained and can drift. Add the `data` term (`--loss data=1,l1=1,l2=1,l3=1`) to constrain it.

Two older terms remain for comparison. Neither helped in our tests.

- **`res`**: $\Phi(d-u)$ against $\langle\Phi(n_i)\rangle$, i.e. what is removed looks like noise.
- **`cross`**: $\Phi(2u-d)$ against $\Phi(d)$, i.e. the residual is uncorrelated with $u$.

## Convolutions on the sphere

The convolutions of the network are those of the scattering statistics: the STL-Dev and FOSCAT stencil ({mod}`stlia.unet`, {mod}`stlia.stl_torch_geometry`).

For every pixel:

1. the $k\times k$ taps of the kernel are laid in the tangent plane, with the pixel size as spacing;
2. they are rotated to the pixel and to its local gauge;
3. each tap is the bilinear interpolation of its four nearest pixels.

The operator is the same everywhere on the sphere: the base faces of HEALPix play no part and there are no edges. It is applied as a sparse gather, followed by a dense product with the kernel.

With a single orientation, the gauge has to be chosen (`--gauge`):

- **`phi`**: the kernel follows the local meridian. It is singular only at the poles.
- **`cosmo`**: the gauge of the STL engine. It is continuous through the poles and flipped between the hemispheres.

The first prototype used planar $3\times3$ convolutions on the twelve base-face images (`--conv faces`). It leaves artefacts along the face edges and is kept only for comparison.

The encoder goes down the HEALPix NESTED pyramid by the mean of the four children of every pixel. The decoder comes back up by copying each parent into its children, with skip connections at every level.

## Training on disc patches

The cost of full-sky statistics grows as $\nside^2$. `stlia-train-patches` trains instead on random discs ({mod}`stlia.regions`):

1. every step draws centres among the pixel centres of a coarse grid (`--centre-nside`) and takes all the pixels within `--radius` of each;
2. the network runs on the discs. At every level, its stencil is bound to the pixels present: taps outside the disc get weight zero, and the weights are renormalised;
3. the losses use the disc statistics of the engine (`PatchEngine`), averaged within `--core` of the centre.

With pixel normalisation and a fixed input scale, the network on a disc equals the network on the full sky wherever its receptive field stays inside the disc. With 3 levels at $\nside=256$, that is within about $4^\circ$ of the centre. The ring between `--core` and `--radius` absorbs the edge effects of the network and of the wavelets.

The disc statistics run on CPU threads, or on the GPU (`--stl-backend gpu`).

## Mock and real data

- **Mock data** (default). The sky $s$ is a Planck PR3 map. Observations add FFP10 noise draws: simulations 10–49 for training and 0–9 for evaluation. The evaluation compares $u$ with $s$.
  
  Because the PR3 map itself contains the Planck noise, this is a test of the method, not a clean-sky reference: $u$ is trained to reproduce the statistics of a sky that is itself noisy.
- **Real data** (`--real`). The observations are the PR3 maps themselves:
  - $d$ is the full-mission map and $d_1, d_2$ the half missions;
  - the noise draws enter only the realisations;
  - both fields are trained;
  - the evaluation reports $\mathrm{rms}(d-u)$ over the noise rms, which is 1 when $u$ removes exactly as much as the noise.
  
  This is the amortised form of the per-map optimisation of Delouis et al. (2022).
