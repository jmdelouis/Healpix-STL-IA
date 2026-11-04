"""Degrade a Planck PR3 map to the working resolutions.

    python prepare_planck.py IN.fits OUTDIR [--name planck353 --fields IQU --scale 1e6 --nside 256 128 64]

Writes OUTDIR/{name}_{field}_n{nside}.npy: NESTED float64 maps, times
`scale` (1e6: K_CMB -> uK_CMB; 1 for the 857 GHz map in MJy/sr, `--fields I`),
each pixel the mean of its children (the parent mean of the pyramid used by
the network and by STL).  Names used: planck353 (full mission),
planck353hm1 / planck353hm2 (half missions), planck857.
"""

import argparse
import os

import healpy as hp
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fits")
    p.add_argument("out")
    p.add_argument("--name", default="planck353")
    p.add_argument("--fields", default="IQU")
    p.add_argument("--scale", type=float, default=1e6)
    p.add_argument("--nside", type=int, nargs="+", default=[256, 128, 64])
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    maps = hp.read_map(a.fits, field=tuple(range(len(a.fields))), nest=True, dtype=np.float64)
    if len(a.fields) == 1:
        maps = [maps]
    for name, m in zip(a.fields, maps):
        bad = ~np.isfinite(m) | (m == hp.UNSEEN)
        m = m * a.scale
        assert not bad.any(), f"{name}: {bad.sum()} unseen pixels"
        for nside in sorted(a.nside, reverse=True):
            while m.size > 12 * nside * nside:
                m = m.reshape(-1, 4).mean(1)
            np.save(os.path.join(a.out, f"{a.name}_{name}_n{nside}.npy"), m)
            print(f"{a.name} {name} nside {nside} mean {m.mean():.4g} std {m.std():.4g}", flush=True)


if __name__ == "__main__":
    main()
