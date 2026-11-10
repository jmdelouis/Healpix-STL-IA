# healpix-stl-ia

[![Test](https://github.com/jmdelouis/Healpix-STL-IA/actions/workflows/test.yml/badge.svg)](https://github.com/jmdelouis/Healpix-STL-IA/actions/workflows/test.yml)
[![Docs](https://github.com/jmdelouis/Healpix-STL-IA/actions/workflows/docs.yml/badge.svg)](https://jmdelouis.github.io/Healpix-STL-IA/)

**Amortised scattering denoising of full-sky HEALPix maps.**

A U-Net on the sphere maps an observed map `d` to a denoised map `u = f_θ(d)`. It is trained without any clean sky: the loss compares only the scattering covariances Φ computed by [STL-Dev-MPI](https://github.com/jmdelouis/STL-Dev-MPI) (for example Φ(u + n_i) against Φ(d)), with the losses of Delouis et al. (2022, A&A 668, A122). The per-map L-BFGS optimisation of that paper becomes one network evaluation.

- **Convolutions on the sphere**: the STL/FOSCAT stencil, with no base-face edges.
- **Full-sky or disc-patch training**, with the disc statistics on CPU or GPU.
- **Mock data** (FFP10 noise draws) or **the Planck PR3 maps themselves** (`--real`).

**Documentation: <https://jmdelouis.github.io/Healpix-STL-IA/>**:
- tutorials, from a first run at nside 64 to the reference at 256 and nside 512/1024;
- every option of the production trainer, with what changing it did in the paper;
- method and API.

The production trainer is `train_native_unet.py` of STL-Dev-MPI, tag `2026.11.01`.

## Install

```bash
pip install torch                       # the CUDA build of your machine
pip install "healpix-stl-ia @ git+https://github.com/jmdelouis/Healpix-STL-IA"
```

Training also needs the STL-Dev-MPI engine (Rust, built from source). See the [installation page](https://jmdelouis.github.io/Healpix-STL-IA/installation.html).

```bash
stlia-train-denoise --data maps --out runs/n64 --nside 64 --noise uniform --loss data=1
stlia-train-denoise --data maps --out runs/real256 --nside 256 --real --loss data=1,l1=1,l2=1,l3=1
stlia-train-patches --data maps --out runs/p256 --nside 256 --stl-backend gpu --device cuda:0
```

## Repository

| path | content |
|---|---|
| `stlia/` | the package: network (`unet`), losses (`stl_torch`), disc patches (`regions`), training (`train_denoise`, `train_patches`) |
| `scripts/` | data preparation, figures, and the HPC job scripts |
| `tests/` | unit tests (`pytest`); the tests marked `stl` need the engine |
| `docs/` | the documentation (Sphinx, MyST); API and command-line pages are generated from the code |
| `text/` | the A&A paper, synchronised with Overleaf |
| `REPRODUCE.md` | step-by-step recipe for every run, figure and table of the paper, on any GPU machine |
| `HANDOFF.md` | state of the project: where the data and runs are, what is finished, open problems, failed attempts |

Data and heavy computations live on an HPC work directory, not in this repository.

## Contributing

Every code change comes with its documentation and a `CHANGELOG.md` entry, and the CI checks this: see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0.
