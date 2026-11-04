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
    p.add_argument("--fill-from", default=None,
                   help="name of an already prepared map (e.g. planck353) for the degraded pixels with no seen child")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    maps = hp.read_map(a.fits, field=tuple(range(len(a.fields))), nest=True, dtype=np.float64)
    if len(a.fields) == 1:
        maps = [maps]
    for name, m in zip(a.fields, maps):
        # Unseen pixels (the half-mission maps have a few): parent mean over the seen children only.
        w = (np.isfinite(m) & (m != hp.UNSEEN)).astype(np.float64)
        m = np.where(w > 0, m * a.scale, 0.0)
        print(f"{a.name} {name}: {int((w == 0).sum())} unseen pixels at nside {hp.npix2nside(m.size)}", flush=True)
        for nside in sorted(a.nside, reverse=True):
            while m.size > 12 * nside * nside:
                mw, w = (m * w).reshape(-1, 4).sum(1), w.reshape(-1, 4).sum(1)
                m = np.where(w > 0, mw / np.maximum(w, 1e-300), 0.0)
                w = (w > 0).astype(np.float64)  # a parent with a seen child is seen
            out = m.copy()
            empty = w == 0
            if empty.any():
                assert a.fill_from, f"{name}: {int(empty.sum())} pixels without seen child at nside {nside}"
                out[empty] = np.load(os.path.join(a.out, f"{a.fill_from}_{name}_n{nside}.npy"))[empty]
            np.save(os.path.join(a.out, f"{a.name}_{name}_n{nside}.npy"), out)
            print(f"{a.name} {name} nside {nside} mean {out.mean():.4g} std {out.std():.4g} "
                  f"(filled from {a.fill_from}: {int(empty.sum())})", flush=True)


if __name__ == "__main__":
    main()
