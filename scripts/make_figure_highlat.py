"""Spectra of the denoised maps of several `--real` runs at high latitude.

    python make_figure_highlat.py DATA_DIR OUT.pdf RUN_DIR:LABEL [RUN_DIR:LABEL ...]

DATA_DIR holds the half-mission maps (planck353hm{1,2}_{Q,U}_n{nside}.npy),
each RUN_DIR eval_{Q,U}.npz (d, u).  Draws, over |b| > 30 and |b| > 60 deg
(rows) for EE and BB (columns), the ratio of the spectrum of u to the
half-mission cross spectrum d1 x d2, which has no noise bias, for each run.
"""

import argparse
import os

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("runs", nargs="+")
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--bcuts", default="30,60")
    p.add_argument("--apod", type=float, default=5.0)
    p.add_argument("--lmin", type=int, default=10)
    p.add_argument("--lmax", type=int, default=600)
    p.add_argument("--bins", type=int, default=14)
    a = p.parse_args()
    hm = [{f: np.load(os.path.join(a.data, f"planck353hm{h}_{f}_n{a.nside}.npy")) for f in "QU"} for h in (1, 2)]
    runs = []
    for r in a.runs:
        path, _, label = r.partition(":")
        ev = {f: np.load(os.path.join(path, f"eval_{f}.npz")) for f in "QU"}
        runs.append((label or os.path.basename(path), {f: ev[f]["u"] for f in "QU"}))
    theta, _ = hp.pix2ang(a.nside, np.arange(12 * a.nside**2))
    edges = np.unique(np.round(np.geomspace(a.lmin, a.lmax + 1, a.bins + 1)).astype(int))
    ell = 0.5 * (edges[:-1] + edges[1:] - 1)
    bcuts = [float(b) for b in a.bcuts.split(",")]

    fig, axes = plt.subplots(len(bcuts), 2, figsize=(10, 3.3 * len(bcuts)), sharex=True, squeeze=False)
    for row, bcut in zip(axes, bcuts):
        mask = (np.abs(90.0 - np.degrees(theta)) > bcut).astype(float)
        mask = np.clip(hp.smoothing(mask, fwhm=np.radians(a.apod)), 0, 1)

        def spec(m1, m2=None):
            pol = lambda m: [np.zeros(m["Q"].size)] + [hp.reorder(m[f], n2r=True) * mask for f in "QU"]  # noqa: E731
            cl = hp.anafast(pol(m1), None if m2 is None else pol(m2), lmax=a.lmax)
            return [np.array([c[i:j].mean() for i, j in zip(edges[:-1], edges[1:])]) for c in cl[1:3]]

        x = spec(hm[0], hm[1])
        for k, (label, u) in enumerate(runs):
            for ax, cu, cx in zip(row, spec(u), x):
                ax.semilogx(ell, cu / cx, "o-", ms=3, color=f"C{k}", label=label)
        for ax, t in zip(row, ("EE", "BB")):
            ax.axhline(1, color="k", lw=0.8)
            ax.set_title(rf"{t}, $|b|>{bcut:.0f}^\circ$")
        row[0].set_ylabel(r"$C_\ell(u)\,/\,C_\ell(d_1\times d_2)$")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\ell$")
        ax.set_xticks([t for t in (10, 30, 100, 300) if a.lmin <= t <= a.lmax], minor=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out)


if __name__ == "__main__":
    main()
