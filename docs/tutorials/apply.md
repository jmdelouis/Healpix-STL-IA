# Apply a trained network

A run writes its weights to `params.npy` (and `model.pt` when torch is importable). Applying them to a map is one
evaluation of the network: a fraction of a second at $\nside=256$ on a GPU. The network has to be rebuilt with the
options of its training: `--nside`, `--channels`, `--cond`, `--pol`. The conditioning maps have to be computed as the
trainer computed them.

## In Python

This is what `scripts/apply_native.py` does for the known sky, written for any pair of $(Q,U)$ maps. The maps are NESTED,
in $\mu$K$_{\rm CMB}$, at the $\nside$ of the training.

```python
import math, os
import numpy as np
from stl_port.denoise import conditioning      # STL-Dev-MPI 2026.11.01, on PYTHONPATH (scripts/env.sh)
from stl_port.native import NativeUNet

run, data, n = "runs/n256_ebqu_c16_cl3n", "data/maps", 256
channels = [16, 32, 48, 64, 64, 64]            # --channels of the training
kinds = ["log857", "sigma"]                    # --cond log857+sigma
sims = range(10, 50)                           # --sims of the training (noise level of "sigma")

load = lambda name: np.load(os.path.join(data, f"{name}_n{n}.npy")).astype(np.float64)
net = NativeUNet(n, channels, cond=len(kinds), j=int(math.log2(n)) - 1, backend="gpu", device=0,
                 maps=2)                       # maps=2: --pol ebqu (or eb, spin); 1 for --pol qu
net.set_params(np.load(os.path.join(run, "params.npy")))
i857 = load("planck857_I")
net.set_common(i857)
noise_var = [np.mean([np.load(os.path.join(data, f"planck353_noise_{f}_n{n}_mc{k}.npy")) ** 2 for k in sims], 0)
             for f in "QU"]
net.set_cond(conditioning(kinds, i857, noise_var))

d = np.stack([load("planck353_Q"), load("planck353_U")])   # the map to denoise, [2, npix]
u = net.apply(d)                                             # the denoised (Q, U), [2, npix]
```

Notes:
- **Where the code comes from:** this is the code of `apply_native.py`, which takes the truth plus a noise simulation
  as `d`.
- **Input scaling:** `net.apply` scales each input map by its own std, as `--input-scale map` did in training.
- **`--pol qu`:** use `maps=1` and apply the network to $Q$ and $U$ separately.

## On other noise realisations of a mock sky

```bash
$PY scripts/apply_native.py RUN SIM_DIR OUT.npz --mcs 1-9 \
    --nside 256 --channels 16,32,48,64,64,64 --cond log857+sigma --sims 10-49 --pol ebqu
```

This writes `OUT_mc{k}.npz` with `Q`, `U` and `mc`, for `eval_truth.py --maps` ({doc}`known-sky`).

## In PyTorch

`model.pt` holds the weights under the names of `stlia.unet.UNet`, with the network description.
`stlia.unet.load(path, device)` reads it, so the network can be used without the engine (the native and PyTorch
networks give the same $u$ to $4\times10^{-7}$). With `--pol ebqu`, the network maps $(Q,U)$ to $(Q,U)$ and the
E/B transforms only enter the loss, so applying it needs no transform. The conditioning maps must be the same as above.
