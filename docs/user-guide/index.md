# PyTorch implementation

The package `stlia` holds the PyTorch network and losses with which the method was developed (the early tests of the paper, items 1-10 of its Sect. 7). It is the reference implementation of the losses, and its network `stlia.unet.UNet` is the one the native trainer saves and loads (`model.pt`). The maps of the paper come from the native trainer: see {doc}`../tutorials/index` and {doc}`../options/native-trainer`.

These pages follow a PyTorch run from the data to the figures:

1. {doc}`data`: download the Planck maps and noise simulations, and degrade them to the working resolutions.
2. {doc}`full-sky`: train on whole maps with `stlia-train-denoise`, on mock data or on the real PR3 maps.
3. {doc}`patches`: train on random disc patches with `stlia-train-patches`, with the disc engine on CPU or GPU.
4. {doc}`outputs`: what a run writes, how to read its evaluations, and how to load a trained network.

```{toctree}
:maxdepth: 1

data
full-sky
patches
outputs
```
