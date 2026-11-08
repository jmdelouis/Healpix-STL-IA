"""A mock PR3 observation of a known sky (make_pysm_sky.py), for the validation.

    python make_sim_obs.py SKY_DIR DATA_DIR OUT_DIR [--nside 256 --mc 0 --match 30]

The sky s (sky353_{Q,U}) is first given the EE and BB spectra of the PR3 data:
each of E and B is multiplied in harmonic space by a power law in ell, fitted to
sqrt(C_l(d1 x d2) / C_l(s)) over 30 <= l < 450 under |b| > --match deg (apodised
by 5 deg), the half-mission cross spectrum of the PR3 maps having no noise bias.
The observation is then d = s + n, d1 = s + n1, d2 = s + n2 with the FFP10
noise simulation --mc (full mission and its two half missions), which must not
be among the training simulations. OUT_DIR gets the files the training drivers
read, under the PR3 names (planck353_{Q,U}, planck353hm{1,2}_{Q,U},
planck857_I = the sky at 857 GHz), symbolic links to the other files of DATA_DIR
at this nside (noise simulations and std maps, I maps), and the truth
truth353_{Q,U}_n{nside}.npy.
"""

import argparse
import glob
import os

import healpy as hp
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sky")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--mc", type=int, default=0)
    p.add_argument("--match", type=float, default=30.0)
    p.add_argument("--lrange", type=int, nargs=2, default=[30, 450])
    a = p.parse_args()
    n = a.nside
    os.makedirs(a.out, exist_ok=True)
    load = lambda d, name: np.load(os.path.join(d, f"{name}_n{n}.npy"))  # noqa: E731
    lmax = 3 * n - 1
    theta, _ = hp.pix2ang(n, np.arange(12 * n * n))
    mask = (np.abs(90 - np.degrees(theta)) > a.match).astype(float)
    mask = np.clip(hp.smoothing(mask, fwhm=np.radians(5.0)), 0, 1)
    w2 = np.mean(mask**2)

    def ring(m):
        return hp.reorder(m, n2r=True)

    def spectra(qu1, qu2=None):
        f = lambda qu: [np.zeros(12 * n * n)] + [ring(x) * mask for x in qu]  # noqa: E731
        cl = hp.anafast(f(qu1), None if qu2 is None else f(qu2), lmax=lmax, iter=3) / w2
        return cl[1], cl[2]

    s = [load(a.sky, f"sky353_{f}") for f in "QU"]
    x = spectra([load(a.data, f"planck353hm1_{f}") for f in "QU"], [load(a.data, f"planck353hm2_{f}") for f in "QU"])
    cs = spectra(s)
    ell = np.arange(lmax + 1)
    sel = (ell >= a.lrange[0]) & (ell < a.lrange[1])
    gains = []
    for name, cx, c in zip("EB", x, cs):
        edges = np.arange(a.lrange[0], a.lrange[1] + 1, 30)
        lb = np.array([ell[i:j].mean() for i, j in zip(edges[:-1], edges[1:])])
        rb = np.array([cx[i:j].mean() / c[i:j].mean() for i, j in zip(edges[:-1], edges[1:])])
        ok = rb > 0
        slope, icpt = np.polyfit(np.log(lb[ok]), 0.5 * np.log(rb[ok]), 1)
        g = np.exp(icpt) * np.maximum(ell, 2) ** slope
        g[:2] = 0
        gains.append(g)
        print(f"{name}: amplitude gain {np.exp(icpt) * 100 ** slope:.3f} at l=100, slope {slope:+.3f}; "
              f"C(s)/C(d1xd2) before over {a.lrange}: {c[sel].sum() / cx[sel].sum():.3f}", flush=True)
    # s + QU((g - 1) EB(s)): the part of s outside the harmonic round trip is kept
    alm = hp.map2alm([np.zeros(12 * n * n)] + [ring(m) for m in s], lmax=lmax, iter=3, pol=True)
    alm[0] = 0
    alm[1] = hp.almxfl(alm[1], gains[0] - 1)
    alm[2] = hp.almxfl(alm[2], gains[1] - 1)
    dq = hp.alm2map(alm, n, lmax=lmax, pol=True)
    truth = [m + hp.reorder(c, r2n=True) for m, c in zip(s, dq[1:])]
    after = spectra(truth)
    for name, cx, c in zip("EB", x, after):
        bands = ((30, 60), (100, 200), (200, 300), (300, 450))
        print(f"{name} after: C(s)/C(d1xd2) " + " ".join(f"{i}-{j} {c[i:j].mean() / cx[i:j].mean():.2f}"
                                                         for i, j in bands), flush=True)
    for f, t in zip("QU", truth):
        np.save(os.path.join(a.out, f"truth353_{f}_n{n}.npy"), t)
        for kind, name in (("", "planck353"), ("hm1", "planck353hm1"), ("hm2", "planck353hm2")):
            noise = np.load(os.path.join(a.data, f"planck353_noise{kind}_{f}_n{n}_mc{a.mc}.npy"))
            np.save(os.path.join(a.out, f"{name}_{f}_n{n}.npy"), t + noise)
        print(f"{f}: rms truth {t.std():.4g}, noise mc{a.mc} {noise.std():.4g} uK", flush=True)
    np.save(os.path.join(a.out, f"planck857_I_n{n}.npy"), load(a.sky, "sky857_I"))
    # everything else the drivers read (noise simulations, noise std maps, I maps): links to DATA_DIR
    for path in glob.glob(os.path.join(a.data, f"*_n{n}*.npy")):
        dst = os.path.join(a.out, os.path.basename(path))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(path), dst)


if __name__ == "__main__":
    main()
