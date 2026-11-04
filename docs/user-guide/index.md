# User guide

These pages follow a run from the data to the figures:

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
