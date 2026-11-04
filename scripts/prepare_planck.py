"""Degrade the Planck PR3 353 GHz map to the working resolutions.

    python prepare_planck.py IN.fits OUTDIR [--nside 256 128 64]

Writes OUTDIR/planck353_{I,Q,U}_n{nside}.npy: NESTED float64 maps in uK_CMB,
each pixel the mean of its children (the parent mean of the pyramid used by
the network and by STL).
"""

import argparse
import os

import healpy as hp
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fits")
    p.add_argument("out")
    p.add_argument("--nside", type=int, nargs="+", default=[256, 128, 64])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    maps = hp.read_map(a.fits, field=(0, 1, 2), nest=True, dtype=np.float64)
    for name, m in zip("IQU", maps):
        m = m * 1e6
        bad = ~np.isfinite(m) | (m == hp.UNSEEN * 1e6)
        assert not bad.any(), f"{name}: {bad.sum()} unseen pixels"
        for nside in sorted(a.nside, reverse=True):
            while m.size > 12 * nside * nside:
                m = m.reshape(-1, 4).mean(1)
            np.save(os.path.join(a.out, f"planck353_{name}_n{nside}.npy"), m)
            print(f"{name} nside {nside} mean {m.mean():.4g} std {m.std():.4g} uK", flush=True)


if __name__ == "__main__":
    main()
