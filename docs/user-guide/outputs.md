# Outputs of a run

Every run writes to its `--out` directory:

| file | content |
|---|---|
| `log` | the command-line summary, one line every `--log-every` steps (loss terms, time per step), and the evaluations |
| `train.json` | the arguments, the number of coefficients and parameters, the time, the loss history and every evaluation (with `--optimizer lbfgs`: per round, the number of loss evaluations and the loss at its start and end) |
| `model.pt` | the network weights and configuration, saved at every evaluation (not written with `--optimizer lbfgs`) |
| `eval_Q.npz`, `eval_U.npz` | the maps of the final evaluation: `sky`, `d`, `u` (NESTED). With `--real`, `sky` is `d` |

## Evaluations

An evaluation line reads:

```
eval 3000 Q: pixel err/noise 1.086 data 0.1217 | u/d vs sky [sigma]: mean 0.644/0.673 var 0.193/1.73 S1 0.94/25.3 ...
```

Its fields are:

- **`pixel err/noise`**: the rms of $u - s$ over the rms of the noise. With `--real`, it is the rms of $d - u$ over the noise rms.
- **the loss terms**: on fixed test noise draws.
- **`u/d vs sky [sigma]`**: for each family of statistics, the rms of $(\Phi_c(u) - \Phi_c(s))/\sigma_c(s)$. Before the slash for $u$, after it for $d$. A value near 1 means $u$ is as close to the sky as one ergodic standard deviation.

`train.json` holds the same numbers under `evals[step][field]`. Its keys are `pixel_rms_error_over_noise`, `loss_terms`, `stats_u_vs_sky`, `stats_d_vs_sky`, and, for `stlia-train-denoise`, `level_power_u_over_sky`, `level_power_d_over_sky`.

## Using a trained network

```python
import numpy as np
import torch
from stlia.unet import load

net, config = load("runs/n256/model.pt", "cuda")
d = torch.as_tensor(np.load("maps/planck353_Q_n256.npy"), device="cuda")
with torch.no_grad():
    u = net(d[None], config.get("scale") or d.std())[0]
```

The input is a NESTED map at the `nside` of the network. Full-sky networks were trained with the standard deviation of each map as input scale; patch networks carry a fixed `scale`.

## Figures

The scripts below read the run directories:

- `scripts/make_figures.py`: the $N_{\rm side}=64$ figures and table of the paper;
- `scripts/make_figures256.py`: the $N_{\rm side}=256$ figures and table;
- `scripts/make_figures_real.py`: the maps, the BICEP2 field and the EE/BB spectra of a `--real` run.

`bash scripts/reproduce.sh figures` runs all three.
