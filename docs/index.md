# healpix-stl-ia

**Denoise full-sky HEALPix maps with a network trained on scattering statistics alone: no clean sky is ever used.**

`healpix-stl-ia` trains a U-Net on the sphere. The network maps an observed map $d$ to a denoised map $u = f_\theta(d)$. The training loss compares only scattering covariances $\Phi$, computed by the [STL-Dev-MPI](https://github.com/jmdelouis/STL-Dev-MPI) engine. In the simplest loss, $u$ is good when the statistics of $u + n_i$, for noise realisations $n_i$, match those of $d$.

This is the statistical denoising of [Delouis et al. (2022)](#delouis2022), which they ran as one L-BFGS optimisation per map. Here it is *amortised*: once trained, the network denoises a map in a single evaluation.

```
d (noisy HEALPix map) ──► spherical U-Net ──► u ──► Φ(u + n_i) ≈ Φ(d)   (loss)
```

The main features are:

- **Convolutions on the sphere.** The network uses the STL/FOSCAT stencil: a $k\times k$ kernel laid in the tangent plane of every pixel and interpolated on the HEALPix grid. The 12 base faces play no part, so there are no face edges.
- **The losses of Delouis et al. (2022).** Besides the auto loss, the package implements the three cross-statistics terms: half-mission maps, $d$ against $u$, and a dust template.
- **Full sky or disc patches.** The network can be trained on whole maps, or on random $10^\circ$ discs with the disc statistics of the engine, on CPU or GPU.
- **Planck PR3 pipeline.** Scripts download, prepare and train on the 353 GHz polarisation maps and the FFP10 noise simulations, with a complete reproduction recipe.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc
The package, the STL-Dev-MPI engine and a GPU environment.
:::

:::{grid-item-card} Method
:link: method
:link-type: doc
Losses, spherical convolutions, pyramid, patches.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc
Data, full-sky and patch training, real data, outputs.
:::

:::{grid-item-card} Reproducing the paper
:link: reproduce
:link-type: doc
Every run, figure and table, step by step, on any GPU machine.
:::

:::{grid-item-card} Command line
:link: cli
:link-type: doc
Every option of `stlia-train-denoise` and `stlia-train-patches`.
:::

:::{grid-item-card} API reference
:link: api
:link-type: doc
The `stlia` modules, generated from the code.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

installation
method
rust_port
user-guide/index
reproduce
cli
api
changelog
contributing
```

(delouis2022)=
## Reference

Delouis, J.-M., Allys, E., Gauvrit, E., Boulanger, F. 2022, *Non-Gaussian modelling and statistical denoising of Planck dust polarisation full-sky maps using scattering transforms*, A&A 668, A122, [doi:10.1051/0004-6361/202244566](https://doi.org/10.1051/0004-6361/202244566).
