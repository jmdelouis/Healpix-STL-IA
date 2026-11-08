"""Apply a trained native U-Net (train_native.slurm, --pol ebqu or qu) to other noise realisations of a mock sky.

    python apply_native.py RUN_DIR SIM_DIR OUT.npz --mcs 1-9 [--nside 256 --channels 16,32,48,64,64,64
        --cond log857+sigma --sims 10-49 --pol ebqu --lib $G/src/python/stl_ensemble]

Rebuilds the network of RUN_DIR (params.npy) with the training options given here, its conditioning
maps as the driver does (857 GHz map of SIM_DIR, noise variance of the training simulations --sims),
and applies it to d_k = truth + n_k for each FFP10 simulation k of --mcs (SIM_DIR/truth353_{Q,U},
SIM_DIR/planck353_noise_{Q,U}_n{nside}_mc{k}). Writes OUT_mc{k}.npz (Q, U, mc) for eval_truth.py --maps.
"""

import argparse
import math
import os
import sys

import numpy as np


def id_range(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("sim")
    p.add_argument("out")
    p.add_argument("--mcs", default="1-9")
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--channels", default="16,32,48,64,64,64")
    p.add_argument("--cond", default="log857+sigma")
    p.add_argument("--sims", default="10-49", help="the training simulations (noise variance of --cond sigma)")
    p.add_argument("--pol", default="ebqu", choices=["qu", "ebqu"])
    p.add_argument("--lib", default=os.path.join(os.environ.get("G", ""), "src/python/stl_ensemble"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--backend", default="gpu")
    a = p.parse_args()
    sys.path.insert(0, a.lib)
    from stl_port.denoise import conditioning
    from stl_port.native import NativeUNet

    n = a.nside
    load = lambda name: np.load(os.path.join(a.sim, f"{name}_n{n}.npy")).astype(np.float64)  # noqa: E731
    channels = [int(c) for c in a.channels.split(",")]
    kinds = [] if a.cond == "none" else a.cond.split("+")
    joint = a.pol != "qu"
    net = NativeUNet(n, channels, cond=len(kinds), j=int(math.log2(n)) - 1, backend=a.backend, device=a.device,
                     maps=2 if joint else 1)
    net.set_params(np.load(os.path.join(a.run, "params.npy")))
    i857 = load("planck857_I")
    net.set_common(i857)
    noise_var = [np.mean([np.load(os.path.join(a.sim, f"planck353_noise_{f}_n{n}_mc{k}.npy")) ** 2
                          for k in id_range(a.sims)], 0) for f in "QU"]
    if kinds:
        net.set_cond(conditioning(kinds, i857, noise_var))
    s = np.stack([load(f"truth353_{f}") for f in "QU"])
    base = a.out[:-4] if a.out.endswith(".npz") else a.out
    for k in id_range(a.mcs):
        d = s + np.stack([np.load(os.path.join(a.sim, f"planck353_noise_{f}_n{n}_mc{k}.npy")) for f in "QU"])
        u = net.apply(d) if joint else np.stack([net.apply(d[i]) for i in range(2)])
        np.savez(f"{base}_mc{k}.npz", Q=u[0], U=u[1], mc=k)
        print(f"mc{k}: rms d-u Q {np.std(d[0] - u[0]):.3f} U {np.std(d[1] - u[1]):.3f} uK", flush=True)


if __name__ == "__main__":
    main()
