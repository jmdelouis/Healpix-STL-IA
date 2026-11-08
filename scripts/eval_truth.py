"""Denoising errors against a known sky (make_sim_obs.py).

    python eval_truth.py SIM_DIR DATA_DIR RUN_DIR [RUN_DIR ...] [--nside 256 --mc 0] [--maps u.npz ...]

For each run (eval_{Q,U}.npz: d, u), or each --maps file (Q, U, with u for the
observation of noise simulation --mc given in the file), compares u with the
truth s (SIM_DIR/truth353_{Q,U}) and d with s, against the FFP10 noise n of the
observation: the pixel rms of u - s over that of n by band of |b|, and over
|b| > 30 and 60 deg (apodised by 5 deg) the EE and BB spectra C(u)/C(s), the
transfer C(u, s)/C(s), the correlation C(u, s)/sqrt(C(u) C(s)) and the
residual C(u - s)/C(n), in bands of ell. Prints one block per run.
"""

import argparse
import os

import healpy as hp
import numpy as np

BANDS = ((10, 30), (30, 60), (60, 100), (100, 200), (200, 300), (300, 450), (450, 600))
LAT = ((0, 20), (20, 30), (30, 45), (45, 60), (60, 90))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sim")
    p.add_argument("data")
    p.add_argument("runs", nargs="*")
    p.add_argument("--maps", nargs="*", default=[])
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--mc", type=int, default=0)
    a = p.parse_args()
    n = a.nside
    s = [np.load(os.path.join(a.sim, f"truth353_{f}_n{n}.npy")) for f in "QU"]
    theta, _ = hp.pix2ang(n, np.arange(12 * n * n), nest=True)
    b = np.abs(90 - np.degrees(theta))
    masks = {}
    for cut in (30, 60):
        m = (hp.reorder(b, n2r=True) > cut).astype(float)
        m = np.clip(hp.smoothing(m, fwhm=np.radians(5.0)), 0, 1)
        masks[cut] = (m, np.mean(m**2))

    def spec(x, y=None, cut=30):
        m, w2 = masks[cut]
        f = lambda qu: [np.zeros(12 * n * n)] + [hp.reorder(v, n2r=True) * m for v in qu]  # noqa: E731
        cl = hp.anafast(f(x), None if y is None else f(y), lmax=min(600, 3 * n - 1), iter=3) / w2
        return cl[1], cl[2]

    def band(cl):
        return np.array([cl[i:j].mean() for i, j in BANDS])

    cs = {cut: [band(c) for c in spec(s, cut=cut)] for cut in masks}

    def report(label, d, u, noise):
        rn = (noise[0] ** 2 + noise[1] ** 2) / 2
        for name, x in (("u", u), ("d", d)):
            r = ((x[0] - s[0]) ** 2 + (x[1] - s[1]) ** 2) / 2
            print(f"{label} {name}-s rms/noise by |b| {LAT}: "
                  + " ".join(f"{np.sqrt(r[(b >= i) & (b < j)].mean() / rn[(b >= i) & (b < j)].mean()):.3f}"
                             for i, j in LAT))
        cn = {cut: [band(c) for c in spec(noise, cut=cut)] for cut in masks}
        for cut in masks:
            cu = [band(c) for c in spec(u, cut=cut)]
            cus = [band(c) for c in spec(u, s, cut=cut)]
            cr = [band(c) for c in spec([u[0] - s[0], u[1] - s[1]], cut=cut)]
            cd = [band(c) for c in spec([d[0] - s[0], d[1] - s[1]], cut=cut)]
            for i, pol in enumerate(("EE", "BB")):
                fmt = lambda v: " ".join(f"{x:.2f}" for x in v)  # noqa: E731
                print(f"{label} |b|>{cut} {pol} ell {BANDS}")
                print(f"    C(u)/C(s) {fmt(cu[i] / cs[cut][i])} | transfer {fmt(cus[i] / cs[cut][i])} | "
                      f"corr {fmt(cus[i] / np.sqrt(np.abs(cu[i] * cs[cut][i])))}")
                print(f"    C(u-s)/C(n) {fmt(cr[i] / cn[cut][i])} | C(d-s)/C(n) {fmt(cd[i] / cn[cut][i])}")

    for run in a.runs:
        ev = {f: np.load(os.path.join(run, f"eval_{f}.npz")) for f in "QU"}
        d = [ev[f]["d"] for f in "QU"]
        u = [ev[f]["u"] for f in "QU"]
        noise = [np.load(os.path.join(a.data, f"planck353_noise_{f}_n{n}_mc{a.mc}.npy")) for f in "QU"]
        report(os.path.basename(run.rstrip("/")), d, u, noise)
    for path in a.maps:
        z = np.load(path)
        mc = int(z["mc"])
        noise = [np.load(os.path.join(a.data, f"planck353_noise_{f}_n{n}_mc{mc}.npy")) for f in "QU"]
        d = [s[0] + noise[0], s[1] + noise[1]]
        report(f"{os.path.basename(path)} mc{mc}", d, [z["Q"], z["U"]], noise)


if __name__ == "__main__":
    main()
