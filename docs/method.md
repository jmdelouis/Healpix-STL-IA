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

**Warm-up.** The first estimate is made at the first output, which still contains the noise: the bias of a noisy map is not the bias of the clean map the training leads to. With `--bias-warmup N`, the first $N$ steps (L-BFGS: rounds) use live biases, and the first frozen estimate is made at an already denoised $u$. `--init RUN/model.pt` starts a network from the weights of a previous run instead, for example a run with live biases.

### Which coefficients have a noise bias

The bias $\langle\Phi_k(u+n)\rangle-\Phi_k(u)$ of a coefficient is exactly zero when the noise enters it only linearly, since the noise has zero mean and the half-mission noises are independent. In the cross statistics of the engine:

- $S_2(a,b)=\langle a_z b_z^*\rangle$ is bilinear;
- $S_3$ is linear in one map and takes the modulus of the other. It is given in both orders: `c1 = 0` takes the modulus of $a$, `c1 = 1` that of $b$;
- $S_1$ (root normalisation) and $S_4$ (moduli of both maps) are not linear.

Hence ({func}`stlia.train_denoise.unbiased`, with $T$ treated as noise-free):

| term | model | no noise bias | noise bias |
|---|---|---|---|
| `data` | $\Phi(u+n)$ | mean | variance, $S_1$–$S_4$ |
| `l1` | $\Phi(u+\tilde n_1, u+\tilde n_2)$ | $S_2$ | $S_1$, $S_3$, $S_4$ |
| `l2` | $\Phi(u+\tilde n, u)$ | $S_2$, $S_3$ with the modulus of $u$ | $S_1$, $S_3$ with the modulus of $u+\tilde n$, $S_4$ |
| `l3` | $\Phi(T, u+\tilde n)$ | $S_2$, $S_3$ with the modulus of $T$ | $S_1$, $S_3$ with the modulus of $u+\tilde n$, $S_4$ |
| `cl` | $C_b(u+\tilde n_1, u+\tilde n_2)$ | all bins | none |

A check with the engine (a Gaussian map at $\nside=16$, white noise of half its rms, 200 draws) agrees. The mean over the draws of the coefficients without bias departs from the noise-free value by 1.0 to 1.2 standard errors (median), against 10 to 18 for $S_1$, 19 for the auto $S_2$, 4 to 7 for $S_4$ and 2.3 to 2.8 for the biased $S_3$.

With `frozen`, $B_k$ is estimated only for the coefficients with a noise bias, and is zero for the others. An estimate there would be zero up to the scatter of the $M$ draws, and subtracting it would only add that scatter. `--bias-all` estimates every coefficient. With `live`, all scattering coefficients are averaged over the draws, which only add scatter to the unbiased ones. The `cl` term never uses draws (see *Spectrum loss*).

## Masked statistics

With `--stat-masks 0,20,30,45`, every scattering coefficient of every term (`data`, `l1`, `l2`, `l3`) enters once per galactic mask $|b| > b_0$ ($b_0 = 0$: the full sky). Each entry has its own ergodic weight, computed from the variance and the number of pixels averaged under that mask (Delouis et al. 2022 used five masks).

- **What changes.** The wavelet fields are those of the full sky; only the averages that define the coefficients are restricted to the mask. At each coarser level, the mask contains the parents of the pixels of the finer one.
- **Why.** The full-sky averages are dominated by the Galactic plane; the masked ones constrain the faint high-latitude sky.
- **Engine.** The masked averages are computed by the STL-Dev-MPI engine (`NativeEngine(..., masks=...)`, from commit 9038114), with their gradient. They cost 16% (auto) to 27% (cross) more than the full-sky statistics for five masks.

## Spectrum loss

The term `cl` asks $u$ to have the power spectrum of the sky ({mod}`stlia.spectra`). Its noise-free target is the half-mission cross spectrum:

$$
C_b(m\,d_1, m\,d_2) \quad\text{vs}\quad C_b(m\,u, m\,u),
$$

for each galactic mask $m$ (`--cl-masks`, $|b|$ above each cut, apodised by `--cl-apod`). Here $C_b$ is the pseudo-spectrum of the masked maps, divided by the mean of the squared mask and binned in multipole (`--cl-lmin`, `--cl-lmax`, `--cl-width`).

Each bin is weighted by the Gaussian variance of the target, $(C_b^{11}C_b^{22} + (C_b^{12})^2)/\nu_b$, where $\nu_b$ is the number of modes of the bin times the effective sky fraction of the mask.

The model has no noise term and no noise bias, with live or frozen biases alike. The half-mission noises are independent with zero mean, so $\langle C_b(u+\tilde n_1, u+\tilde n_2)\rangle = C_b(u,u)$ exactly. Noise draws would only add scatter to the gradient. An estimated bias, close to zero but noisy, would make the solution drift.

This term constrains the power at high Galactic latitude. The full-sky ergodic weights of the scattering statistics hardly do so, because the Galactic plane dominates their variance.

The spectra come from one of two differentiable `anafast` (`--cl-backend`), which agree with each other and with `healpy.anafast` to about $10^{-12}$:

- **`stl`**: the `SpectrumEngine` of STL-Dev-MPI (from commit 5648ba1), on the CPU or the GPU, with its exact VJP. `auto`, the default, uses it when the installed engine has it.
- **`torch`**: a ring-FFT spherical harmonic transform ({class}`stlia.sht.HEALPixSHT`), adapted from [healpix-analyse](https://github.com/GRID4EARTH/healpix-analyse) and FOSCAT, with gradients through `torch.autograd`.

On a T4 at $\nside=256$, the spectra and gradient of 4 pairs of maps under 3 masks take 1.0 s with the engine, against 16 s with `torch`. By default, they are the spectra of the scalar maps, Q and U separately, as the network treats them. With a joint (Q, U) network (`--joint`, below), `--cl-spin 2` uses the EE and BB cross spectra instead, from the spin-2 transforms of the engine (STL-Dev-MPI 92f2aef or later).

### Cross spectrum with the dust template, and spectrum of the removed part

Two further terms use the same spectra, masks and bins. Both are spectra of $u$ alone, with no noise draw and no bias, live or frozen.

- **`tcl`**: the cross spectrum of the 857 GHz map $T$ with $u$, against that with $d$,
  $C_b(m\,T, m\,d)$ vs $C_b(m\,T, m\,u)$.
  The 353 GHz noise is independent of $T$, so $\langle C_b(T, d)\rangle = C_b(T, s)$: the target has no noise bias. It is weighted by its Gaussian variance, $(C_b^{TT}C_b^{dd} + (C_b^{Td})^2)/\nu_b$. On Q and U it is the TQ and TU cross spectrum. With `--pol-stats eb`, the fields are the E and B maps, and it is TE and TB.
- **`rcl`**: the spectrum of the removed part against that of the noise,
  $C_b(m\,(d-u))$ vs $\langle C_b(m\,n)\rangle$,
  which a perfect denoising satisfies at every scale. On real data, the noise spectrum is that of $d$ itself, $C_b(d,d)-C_b(d_1,d_2)$: the signal and the signal-noise terms cancel in the difference. With the target of `cl`, the two terms then add up to $C_b(d,d)$, so that together they ask $u$ and $d-u$ to be uncorrelated. The other two estimates do not agree with it. On the PR3 maps at $\nside=64$, it is 0.87--0.92 times the spectrum of the half-mission difference $(d_1-d_2)/2$ for $\ell\ge30$ over $|b|>30^\circ$, and 0.77--0.92 times that of the FFP10 simulations. With the half-mission difference as target, `rcl` and `cl` conflict, and the network removes power that is not the noise. Over the full sky, $C_b(d,d)-C_b(d_1,d_2)$ is dominated by the residuals of the Galactic plane at large scales: 3.9 times the half-mission difference on Q for $10\le\ell<40$, and negative on U. `rcl` therefore has its own masks and first multipole, `--rcl-masks` (default `30,45,60`) and `--rcl-lmin` (default 40); the other spectrum options are those of `cl`. On mock data, the target is the mean over the noise model. Each bin is weighted by the Gaussian variance of one realisation of the noise, $2C_b^2/\nu_b$.

- **`xcl`**: the cross spectrum of the denoised map and the removed part against zero, $C_b(m\,u, m\,(d-u))$ vs 0, which a perfect denoising satisfies since the noise is independent of the sky. It needs no target. With `cl` alone, $u$ can have the spectrum of the sky while keeping part of the noise and losing as much signal: $C(d)=C(u)+C(d-u)+2C(u,d-u)$, and on the PR3 maps at $
side=64$ the joint E/B network trained with `cl` removes only 0.37 times the noise rms, with $u$ following the cross spectrum to 1\%. The variance is fixed by the data, $|C_b(d_1,d_2)|\,C_b(h,h)/
u_b$ with $h=(d_1-d_2)/2$, on the masks of `cl`. A variance taken at the current maps vanishes with $d-u$, and holds the network at $u=d$, which satisfies `xcl` exactly: only `cl` excludes it, and with a weight of 10 or more `xcl` keeps the network there. With a weight of 1 the network removes 0.51 times the noise rms, the spectrum of $d-u$ is 0.9 times that of the half-mission difference at $100\le\ell<128$ (against 0.5), and $u$ still follows the cross spectrum to 1\% over $|b|>30^\circ$ (`--real` only).

`--input-noise` $\alpha$ gives the network $d+\alpha n$ at every training step, $n$ a training simulation of the noise (the same for $Q$ and $U$, random sign), while the terms compare $u$ with $d$: then $u=d$ no longer satisfies `xcl`. At $
side=64$, with $\alpha=0.5$, it did not remove more noise than `xcl` alone in 400 steps.

`rcl` is the spectral version of the scattering term `res`, $\Phi(d-u)$ against $\langle\Phi(n)\rangle$. With frozen biases, `res` also takes the statistics of the half-mission difference as its target on real data. These two terms can replace `l2`, $\Phi(d,u)$ against $\langle\Phi(u+\tilde n,u)\rangle$, whose model is biased on $S_1$, $S_4$ and one order of $S_3$, and needs noise draws or frozen biases. The targets of `res` and `rcl` need neither. `tcl`, `rcl` and `xcl` need scalar spectra: `--cl-spin 0`, and `--pol-stats qu` or `eb`.

## A joint (Q, U) network

With `--joint` (`--real` only), one network ({class}`~stlia.unet.UNet` with `maps=2`) takes the pair $(Q, U)$ and returns $(u_Q, u_U)$, with one input scale for both. Every step trains both fields: the scattering terms of each are summed, and the EE/BB spectrum term constrains the pair. Each field differentiates its terms in turn, because the engines keep the state of their last forward only. The gradients add up on the output, and go back through the network once. With `--optimizer lbfgs`, the same terms are minimised over the pixels of the pair $(u_Q, u_U)$, with `--pol-stats qu` or `eb`: the per-map reference of the joint network. Frozen biases are then re-estimated at the start of every round after `--bias-warmup` rounds.

### Statistics of the E and B maps

$Q$ and $U$ are defined in a frame of reference on the sphere, and their scattering statistics taken separately depend on it; those of the E and B maps do not. FOSCAT computes its spin-2 statistics with wavelets built from spin-2 harmonics, which give the wavelet coefficients of E and B directly from $(Q, U)$. On the full sky this is the scalar scattering of the E and B maps. With `--pol-stats eb` (`--joint`), every term is computed on

$$ (E, B) = \mathcal{T}(Q, U),\qquad \mathcal{T}:\ (Q,U)\xrightarrow{\ \text{spin-2 analysis}\ }(a^E_{\ell m}, a^B_{\ell m})\xrightarrow{\ \text{synthesis}\ }(E, B), $$

({class}`~stlia.polar.EBTransform`, the transforms of {mod}`stlia.sht`, to $\ell_{\max}=3N_{\rm side}-1$, as healpy's `map2alm(pol=True)` and `alm2map`). The network still maps $(Q, U)$ to $(Q, U)$; $\mathcal{T}$ is applied to its output, with the gradient through it, and once to the observations and to every noise simulation (the Q and U noises of the same simulation together). $\mathcal{T}$ is linear, so a noise added to $(Q, U)$ is a noise added to $(E, B)$: the noise biases (Table above) are unchanged. With `cl` (`--cl-spin 0`), the spectrum term is then on the spectra of the E and B maps, i.e. the EE and BB spectra. The spin-2 harmonic tables come from the three-term recurrence of the Wigner $d^\ell_{m,\mp2}(\theta)$ in $\ell$, rescaled per ring against underflow, built once per run; they take a few GB of GPU memory at $N_{\rm side}=256$. These transforms are exact but global: every E or B pixel depends on the whole sky, which rules out masks with holes, patches and high resolutions.

### Local spin-2 statistics: $\bar\eth^2$ in the engine

With `--pol-stats chi` (`--joint`), the statistics come from the spin-2 engine of STL-Dev-MPI (`NativeEngine(spin=2)`, branch `claude/spin-eth`). It computes

$$ \chi_E + i\chi_B = \bar\eth^2 (Q + iU), $$

a local spin-2 derivative, exactly frame-independent: harmonically, $\chi_{E,B}$ is $E, B$ weighted by $\sqrt{(\ell+2)!/(\ell-2)!}\simeq\ell^2$. On every pixel, the stencil uses the pixels within two rings, with $(Q,U)$ parallel-transported to the centre, and weights exact on polynomials of degree 4. The resulting E→B leakage is 0.5 %, and $\chi_E$ is within 1.3 % of the exact $\bar\eth^2$ on a smooth field. $\bar\eth^2$ is applied at the finest level only. The engine's scalar pyramid then carries $\chi_E$ and $\chi_B$, and the gradient goes back through it, then through the transpose of the stencil. Within the narrow band of each wavelet, the $\ell^2$ weighting is nearly a constant, which the per-coefficient ergodic normalisation absorbs.

Against the same engine applied to the exact $\chi$ (spin-2 SHT), the statistics agree to:
- 0.4–3 % on $S_1$ and $S_2$ for $j\ge1$ (nside 64, also on the Planck 353 maps);
- 6–14 % on $S_3$ and $S_4$ with $j_1>0$.

The finest scale $j=0$ is off by 30–50 % at the pixel scale, where the SHT reference itself is off by 12 %. The stencil is local, so masks with holes work: with masks eroded by the stencil radius, a 15° hole changes the masked $S_1$ and $S_2$ by less than 0.03 %. The cross statistics are also on $\chi$ (STL-Dev-MPI f4cf030), each for E and for B, without E×B terms:
- `l1` takes $S(\chi(u+\tilde n_1), \chi(u+\tilde n_2))$ and `l2` takes $S(\chi(u+\tilde n), \chi(u))$, both pairs of spin-2 fields;
- `l3` takes $S(T, \chi(u+\tilde n))$, with the 857 GHz map $T$ scalar.

Against the scalar cross engine on the exact $\chi$, at nside 64, they agree to 0.5–2 % on $S_1$ and $S_2$ for $j=1$–3, and to 3–4 % on $S_3$ and $S_4$ for $j_1>0$. With masks, the erosion applies at every level; a coarse level that it would empty keeps the parents of the pixels kept at the level below. The spectrum term is on the EE and BB cross spectra (`--cl-spin 2`).

## Conditioning on the dust intensity

With `--cond log857`, the network also receives $\log I_{857}$, standardised: the 857 GHz intensity, a tracer of the dust and hence of the local signal-to-noise ratio. This map is concatenated:

- to the input of the network;
- averaged down the pyramid, to the input of every decoder level.

The decoding, i.e. how much is removed, can then depend on the local dust intensity. The untrained network is still the identity.

The 857 GHz map is also the template of `l3`. Check that the network does not copy its morphology into $u$: the residual $d-u$ must remain uncorrelated with it.

With `--cond sigma`, the conditioning map is the log of the local noise level, $\tfrac12\log\langle n_Q^2+n_U^2\rangle/2$ over the training simulations, standardised: the network then knows the signal-to-noise ratio from the noise side, which the scanning strategy sets. `--cond log857+sigma` gives both maps.

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

### On the PR3 maps

`--real` trains on the PR3 maps with the terms of the full-sky runs, computed on every disc ({mod}`stlia.patch_real`): `data`, `l1`, `l2`, `l3` on the disc statistics, and `cl` on binned pseudo-spectra of the disc. The options are those of `stlia-train-denoise`: `--joint`, `--pol-stats {qu,eb}`, `--cond`, `--bias {live,frozen}` (a frozen bias is kept per disc and re-estimated every `--bias-every` steps), `--bias-all`, `--cl-lmin`, `--cl-lmax`, `--cl-width`.

With `--pol-stats eb`, the disc maps of (Q, U) are multiplied by a cosine taper, equal to 1 in the core and to 0 at the edge, and turned into E and B by spherical harmonics on the full sky, zero outside the disc. The spectra of `cl` are those of the same tapered maps, divided by the mean square of the taper.

`--centre-nside 2` puts the centres on the 48 cells of nside 2: with `--radius 27 --core 17`, the cores cover the sky. The evaluation runs the trained network on the full sky.

At nside 64 (`pt64_cells`, 1000 steps, joint E/B, frozen biases), the spectra of u stay those of the data within 2 % at |b| > 30 deg. Below ℓ = 60, however, the network removes almost no noise: 3 to 8 % of the noise power, against 50 to 80 % for the same network trained on the full sky (`t64_jebl3p`). A disc of 27 deg does not constrain the large scales.

`stlia-train-denoise --norm pixel` gives the full-sky network the normalisation of the disc network, as a control.

## Mock and real data

- **Mock data** (default). The sky $s$ is a Planck PR3 map. Observations add FFP10 noise draws: simulations 10–49 for training and 0–9 for evaluation. The evaluation compares $u$ with $s$.
  
  Because the PR3 map itself contains the Planck noise, this is a test of the method, not a clean-sky reference: $u$ is trained to reproduce the statistics of a sky that is itself noisy.
- **Real data** (`--real`). The observations are the PR3 maps themselves:
  - $d$ is the full-mission map and $d_1, d_2$ the half missions;
  - the noise draws enter only the realisations;
  - both fields are trained;
  - the evaluation reports $\mathrm{rms}(d-u)$ over the noise rms, which is 1 when $u$ removes exactly as much as the noise.
  
  This is the amortised form of the per-map optimisation of Delouis et al. (2022).
