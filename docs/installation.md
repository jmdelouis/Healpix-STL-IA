# Installation

`healpix-stl-ia` has two parts:

- **the Python package `stlia`**: the network, the losses and the training commands. It installs with `pip`.
- **the scattering engine of STL-Dev-MPI**: a Rust library with a Python wrapper. It is built from source.

The package imports and its unit tests run without the engine. Training needs the engine, and in practice a GPU.

## The Python package

```bash
pip install torch            # pick the CUDA build of your machine, see pytorch.org
pip install "healpix-stl-ia @ git+https://github.com/jmdelouis/Healpix-STL-IA"
```

To work on the code, clone the repository and install it in editable mode, with the test and documentation tools:

```bash
git clone https://github.com/jmdelouis/Healpix-STL-IA
cd Healpix-STL-IA
pip install -e ".[test,docs]"
pytest
```

The package requires Python 3.10 or later, numpy, scipy, torch, healpy and matplotlib. It provides two commands, `stlia-train-denoise` and `stlia-train-patches`. The same programs are `python train_denoise.py` and `python train_patches.py` from the repository root.

## The STL-Dev-MPI engine

The statistics, and their exact gradients, come from the `stl-bridge` library of [STL-Dev-MPI](https://github.com/jmdelouis/STL-Dev-MPI). It is called from Python through `stl_port.native`.

```bash
git clone https://github.com/jmdelouis/STL-Dev-MPI stl/src
git -C stl/src checkout 20329e1d2aeb6b822e194f671367e9eed8eb8193
export CUDARC_CUDA_VERSION=12080      # CUDA version of the driver, from nvidia-smi (12.8 -> 12080)
cd stl/src && cargo build --release -p stl-bridge --features gpu
```

The build needs a stable Rust toolchain (`rustup`; the paper used 1.99.1). It needs neither MPI nor libclang. Then point Python to the library and to its wrapper:

```bash
export STL_NATIVE_LIBRARY=$PWD/target/release/libstl_bridge.so
export PYTHONPATH=$PWD/python/stl_ensemble:$PYTHONPATH
python -c "from stl_port.native import NativeEngine as E; print(len(E(64, 5, batch=1, backend='gpu').plan))"   # 2522
```

At run time, the GPU engine compiles its kernels with NVRTC. A `libnvrtc` matching the driver must be on `LD_LIBRARY_PATH`. It comes with the CUDA toolkit module of a cluster, or with the PyPI wheel `nvidia-cuda-nvrtc-cu12`.

With the engine installed, the tests marked `stl` also run. One of them checks the torch gradient of the statistics against finite differences.

## A complete environment

For a cluster, `scripts/env.sh` sets everything from a few variables: the work directory, the Python interpreter, a site file for modules, and the Slurm options. {doc}`reproduce` describes it, and `scripts/hpc_env.sh` gives the settings of the cluster used for the paper.
