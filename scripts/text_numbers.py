"""Numbers of the text for a PR3 run: removed rms per latitude band, its split between the E/B maps and P,
and spectrum ratios.   python scripts/text_numbers.py RUN DATA [NSIDE]"""
import os
import sys

import healpy as hp
import numpy as np

run, data = sys.argv[1], sys.argv[2]
ns = int(sys.argv[3]) if len(sys.argv) > 3 else 256
lmax = 3 * ns - 1
F = ("Q", "U")
ev = {f: np.load(os.path.join(run, f"eval_{f}.npz")) for f in F}
d = {f: ev[f]["d"] for f in F}
u = {f: ev[f]["u"] for f in F}
r = {f: d[f] - u[f] for f in F}
hm = {h: {f: np.load(os.path.join(data, f"planck353hm{h}_{f}_n{ns}.npy")) for f in F} for h in (1, 2)}
noise = [{f: np.load(os.path.join(data, f"planck353_noise_{f}_n{ns}_mc{k}.npy")) for f in F} for k in range(10)]
sn = {f: np.sqrt(np.mean([n[f] ** 2 for n in noise])) for f in F}
print("sigma(d-u)/sigma_n full sky:", " ".join(f"{f} {r[f].std() / sn[f]:.3f}" for f in F))
th, _ = hp.pix2ang(ns, np.arange(12 * ns * ns), nest=True)
b = np.abs(90 - np.degrees(th))
for lo, hi in ((0, 20), (20, 30), (30, 45), (45, 60), (60, 90)):
    s = (b >= lo) & (b < hi)
    nb = {f: np.sqrt(np.mean([n[f][s] ** 2 for n in noise])) for f in F}
    print(f"band {lo}-{hi}:", " ".join(f"{f} {np.sqrt(np.mean(r[f][s] ** 2)) / nb[f]:.3f}" for f in F))


def roundtrip(m):
    """T^-1 T on a NESTED (Q, U) pair: spin-2 analysis (iter 3) and synthesis."""
    qu = [np.zeros(m["Q"].size)] + [hp.reorder(m[f], n2r=True) for f in F]
    alm = hp.map2alm(qu, lmax=lmax, iter=3, pol=True)
    alm[0] *= 0
    out = hp.alm2map(alm, ns, lmax=lmax, pol=True)
    return {f: hp.reorder(out[i + 1], r2n=True) for i, f in enumerate(F)}


t = roundtrip(r)
tn = roundtrip(noise[0])
for f in F:
    print(f"{f}: E/B part {t[f].std() / sn[f]:.3f} P part {(r[f] - t[f]).std() / sn[f]:.3f} "
          f"(P of noise mc0 {(noise[0][f] - tn[f]).std() / noise[0][f].std():.3f})")

thr, _ = hp.pix2ang(ns, np.arange(12 * ns * ns))


def maskof(bcut):
    m = (np.abs(90.0 - np.degrees(thr)) > bcut).astype(float)
    return np.clip(hp.smoothing(m, fwhm=np.radians(5.0)), 0, 1)


def spec(mask, m1, m2=None):
    def pol(m):
        return [np.zeros(m["Q"].size)] + [hp.reorder(m[f], n2r=True) * mask for f in F]
    cl = hp.anafast(pol(m1), None if m2 is None else pol(m2), lmax=lmax) / np.mean(mask ** 2)
    return cl[1], cl[2]


bins = ((10, 30), (30, 60), (60, 100), (100, 160), (160, 512)) + (((512, 1024), (1024, 3 * ns)) if ns >= 512 else ())
for bcut in (30, 60):
    mk = maskof(bcut)
    nn = [spec(mk, n) for n in noise]
    nn = [np.mean([x[i] for x in nn], 0) for i in range(2)]
    cr, cu, cx = spec(mk, r), spec(mk, u), spec(mk, hm[1], hm[2])
    print(f"|b|>{bcut} (d-u)/n EE:", " ".join(f"{i}-{j} {cr[0][i:j].mean() / nn[0][i:j].mean():.2f}" for i, j in bins))
    print(f"|b|>{bcut} (d-u)/n BB:", " ".join(f"{i}-{j} {cr[1][i:j].mean() / nn[1][i:j].mean():.2f}" for i, j in bins))
    print(f"|b|>{bcut} u/x EE/BB:", " ".join(f"{i}-{j} {cu[0][i:j].mean() / cx[0][i:j].mean():.2f}/"
                                            f"{cu[1][i:j].mean() / cx[1][i:j].mean():.2f}"
                                            for i, j in ((10, 30), (30, 60), (100, 200), (200, 300), (300, 450),
                                                         (450, 600), (600, 768)) + (((768, 1024), (1024, 1280), (1280, 3 * ns)) if ns >= 512 else ())))
