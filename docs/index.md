# healpix-stl-ia

**Denoise full-sky HEALPix maps with a network trained on scattering statistics alone: no clean sky is ever used.**

`healpix-stl-ia` trains a U-Net on the sphere. The network maps an observed map $d$ to a denoised map
$u = f_\theta(d)$. The training loss compares only scattering covariances $\Phi$, computed by the
[STL-Dev-MPI](https://github.com/jmdelouis/STL-Dev-MPI) engine. In the simplest loss, $u$ is good when the statistics
of $u + n_i$, for noise realisations $n_i$, match those of $d$.

This is the statistical denoising of [Delouis et al. (2022)](#delouis2022), which they ran as one L-BFGS optimisation
per map. Here it is *amortised*: once trained, the network denoises a map in a single evaluation.

```
d (noisy HEALPix map) ──► spherical U-Net ──► u ──► Φ(u + n_i) ≈ Φ(d)   (loss)
```

The production code is the native trainer of STL-Dev-MPI, tag **`2026.11.01`**. The network, the statistics, their
exact gradient and the optimiser all run on one GPU, in Rust. This repository provides:
- the scripts (data, evaluation, figures);
- the PyTorch reference implementation `stlia`;
- the paper.

Its main features:

- **Convolutions on the sphere.** A $3\times3$ kernel is laid in the tangent plane of every pixel (the STL/FOSCAT
  stencil), so the base faces of HEALPix play no part and there are no edges.
- **The losses of Delouis et al. (2022)** on the E and B maps of the output, with frozen noise biases, galactic masks
  and a term on the half-mission cross spectrum.
- **Planck PR3.** The 353 GHz polarisation maps are denoised at $\nside=256$ and 512, and the method is validated on a
  known sky.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
From the installation to a denoised map at nside 64, the reference at 256, the known-sky validation, applying a
network, nside 512 and 1024.
:::

:::{grid-item-card} Options
:link: options/index
:link-type: doc
Every option of the production trainer and of the scripts, with its default, the reference value and what changing it
did.
:::

:::{grid-item-card} Method
:link: method
:link-type: doc
Losses, noise biases, spherical convolutions, pyramid.
:::

:::{grid-item-card} Results: the paper
:link: results
:link-type: doc
The experiments: what was measured, run by run.
:::

:::{grid-item-card} Installation
:link: installation
:link-type: doc
The package, the STL-Dev-MPI engine and a GPU environment.
:::

:::{grid-item-card} Reproducing the paper
:link: reproduce
:link-type: doc
Every run, figure and table, step by step.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting started

installation
tutorials/index
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

options/index
method
results
reproduce
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: PyTorch implementation

user-guide/index
api
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Project

changelog
contributing
```

(delouis2022)=
## Reference

Delouis, J.-M., Allys, E., Gauvrit, E., Boulanger, F. 2022, *Non-Gaussian modelling and statistical denoising of
Planck dust polarisation full-sky maps using scattering transforms*, A&A 668, A122,
[doi:10.1051/0004-6361/202244566](https://doi.org/10.1051/0004-6361/202244566).
