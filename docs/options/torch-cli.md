# PyTorch commands: `stlia-train-denoise`, `stlia-train-patches`

The PyTorch trainers of the package `stlia` served for the early tests of the paper (items 1-10 of its Sect. 7) and remain the reference implementation of the losses: the native trainer reproduces their loss and gradient to $10^{-7}$. For production runs use {doc}`native-trainer`. This page is generated from the argument parsers of the code, so it always lists the current options and defaults.

## stlia-train-denoise

Full-sky training (also `python train_denoise.py`); see {doc}`../user-guide/full-sky`.

```{eval-rst}
.. argparse::
   :module: stlia.train_denoise
   :func: parser
   :prog: stlia-train-denoise
```

## stlia-train-patches

Training on disc patches (also `python train_patches.py`); see {doc}`../user-guide/patches`.

```{eval-rst}
.. argparse::
   :module: stlia.train_patches
   :func: parser
   :prog: stlia-train-patches
```
