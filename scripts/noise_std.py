"""Per-pixel noise level from Planck PR3 (FFP10) noise simulations.

    python noise_std.py OUTDIR SIM.fits [SIM.fits ...] [--name noise --nside 256 128 64]

Every simulation is degraded like the sky (`prepare_planck.py`: NESTED, uK_CMB,
parent means) and saved as OUTDIR/planck353_{name}_{I,Q,U}_n{nside}_mc{k}.npy,
`k` the rank of the file in the argument list (pass them sorted by
realisation); the per-pixel standard deviation over the simulations
(ddof = 1) is saved as OUTDIR/planck353_{name}_std_{I,Q,U}_n{nside}.npy.
Names used: noise (full mission), noisehm1 / noisehm2 (half missions).  Degrading the simulations
before taking the std keeps the pixel-to-pixel correlations of the PR3 noise
in the level of the degraded pixels.
"""

import argparse
import os

import healpy as hp
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("sims", nargs="+")
    p.add_argument("--name", default="noise")
    p.add_argument("--nside", type=int, nargs="+", default=[256, 128, 64])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acc = {}
    for k, path in enumerate(a.sims):
        maps = hp.read_map(path, field=(0, 1, 2), nest=True, dtype=np.float64)
        for name, m in zip("IQU", maps):
            m = m * 1e6
            assert np.isfinite(m).all() and not (m == hp.UNSEEN * 1e6).any(), f"{path} {name}: unseen pixels"
            for nside in sorted(a.nside, reverse=True):
                while m.size > 12 * nside * nside:
                    m = m.reshape(-1, 4).mean(1)
                np.save(os.path.join(a.out, f"planck353_{a.name}_{name}_n{nside}_mc{k}.npy"), m)
                s, s2 = acc.get((name, nside), (0.0, 0.0))
                acc[(name, nside)] = (s + m, s2 + m * m)
        print(f"{path} done", flush=True)
    n = len(a.sims)
    for (name, nside), (s, s2) in sorted(acc.items()):
        std = np.sqrt(np.maximum(s2 - s * s / n, 0) / (n - 1))
        np.save(os.path.join(a.out, f"planck353_{a.name}_std_{name}_n{nside}.npy"), std)
        print(f"{name} nside {nside} std: median {np.median(std):.4g} min {std.min():.4g} max {std.max():.4g} uK", flush=True)


if __name__ == "__main__":
    main()
