# Tutorials

Each tutorial gives the commands to run and what they should print. They use the production trainer of STL-Dev-MPI
`2026.11.01` ({doc}`../options/native-trainer`) and need one NVIDIA GPU. Start with the quickstart: it builds everything
the others reuse.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Quickstart: denoise the PR3 maps at nside 64
:link: quickstart
:link-type: doc
Engine, data and a first training in about an hour on one T4 (15 min of training).
:::

:::{grid-item-card} The reference at nside 256
:link: pr3-reference
:link-type: doc
The run of the paper on the PR3 353 GHz maps: training, checks, figures, numbers.
:::

:::{grid-item-card} Validation on a known sky
:link: known-sky
:link-type: doc
A PySM dust sky with the PR3 noise: train, apply to unseen noises, measure against the truth.
:::

:::{grid-item-card} Apply a trained network
:link: apply
:link-type: doc
Load the weights of a run and denoise another map in Python.
:::

:::{grid-item-card} nside 512 and 1024
:link: high-resolution
:link-type: doc
Data, memory options and the spike guard for the high resolutions.
:::

:::{grid-item-card} Reproduce the paper
:link: ../reproduce
:link-type: doc
Every run, figure and table of the paper, step by step.
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

quickstart
pr3-reference
known-sky
apply
high-resolution
```
