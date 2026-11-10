# Options

Every option you may want to change, with its default, the value of the reference run of the paper, and what was
measured when it was changed. The paper is the record of these experiments ({doc}`../results`); these pages say which
option each experiment corresponds to.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Production trainer
:link: native-trainer
:link-type: doc
`train_native_unet.py` of STL-Dev-MPI `2026.11.01`: network, polarisation, loss terms, spectrum term, noise biases,
optimiser and spike guard, memory, evaluation, outputs.
:::

:::{grid-item-card} Scripts
:link: scripts
:link-type: doc
Data download and preparation, known sky, applying a trained network, measures against the truth, figures and the
numbers of the paper.
:::

:::{grid-item-card} PyTorch commands
:link: torch-cli
:link-type: doc
`stlia-train-denoise` and `stlia-train-patches`, generated from the code: the reference implementation of the losses,
disc patches, per-map L-BFGS.
:::
::::

## What to try first

These choices are open, and the options already exist:

| question | options | what is known |
|---|---|---|
| shorter trainings | `--steps`, `--bias-warmup`, `--bias-every` | the loss is within 4 % of its final value at half the steps at $\nside=64$; the live warm-up is 37 % of the time there. Not checked on the spectra. |
| the excess of $u$ above $\ell=2\nside$ | `--cl-lmax`, `--cl-width`, `--cl-masks`, the weight of `cl` in `--loss` | reduced from 1.5/2.2 to 1.06/1.22 (EE/BB, $600\le\ell<768$) by `--cl-lmax 3nside-1` at 256; still 2.5/7 for $1024\le\ell<1280$ at 512 |
| the BB excess in the faint sky | `--stat-masks`, `--cl-masks`, loss weights | 1.56 for $300\le\ell<450$ at $\vert b\vert>60^\circ$ at 256, also with the per-map L-BFGS: it comes from the loss |
| stability at high resolution | `--spike-guard`, `--lr` | needed at 512; the cause of the spikes is not known |
| several seeds | `--seed` | the spread between trainings is a large fraction of the noise in the maps, 0.02 in the spectra |
| nside 1024 | `--nside 1024 --cl-lmax 3071 --live-batch 4 --bias-batch 2 --noise-pools host --spike-guard 3` | fits on an 80 GB A100 (50.5 GB); never trained |

```{toctree}
:hidden:
:maxdepth: 1

native-trainer
scripts
torch-cli
```
