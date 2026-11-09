"""Figures of the validation on a known sky (make_sim_obs.py, train_native.slurm --data SIM_DIR, apply_native.py).

    python make_figures_sim.py SIM_DIR RUN_DIR OUT_DIR [--nside 256 --mc 0 --apply 'RUN_DIR/apply/u_mc*.npz']

RUN_DIR holds eval_{Q,U}.npz (d, u) of the training observation d = s + n_mc; SIM_DIR the truth
truth353_{Q,U} and the noise simulations. Writes OUT_DIR/sim_maps.pdf (full sky, Q and U: the truth s,
the observation d, the denoised u, the error u - s and the noise d - s) and sim_spectra.pdf (EE and BB
over |b| > 30 and 60 deg: s, d, u, the error u - s, its mean over the --apply maps, and the noise).
"""

import argparse
import glob
import os

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sim")
    p.add_argument("run")
    p.add_argument("out")
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--mc", type=int, default=0)
    p.add_argument("--apply", default=None, help="glob of apply_native.py outputs (Q, U, mc) on other noises")
    p.add_argument("--apod", type=float, default=5.0)
    p.add_argument("--lmax", type=int, default=600)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = a.nside
    qu = ("Q", "U")
    s = {f: np.load(os.path.join(a.sim, f"truth353_{f}_n{n}.npy")) for f in qu}
    ev = {f: np.load(os.path.join(a.run, f"eval_{f}.npz")) for f in qu}
    d = {f: ev[f]["d"] for f in qu}
    u = {f: ev[f]["u"] for f in qu}
    noise = {f: d[f] - s[f] for f in qu}
    err = {f: u[f] - s[f] for f in qu}

    # Full sky: s, d, u with the same histogram-equalised colours (those of s), then u - s and d - s
    # on one linear scale, that of the noise.
    fig = plt.figure(figsize=(26, 7.4))
    for row, f in enumerate(qu):
        lo, hi = np.percentile(s[f], [0.5, 99.5])
        v = np.percentile(np.abs(noise[f]), 99)
        panels = [(s[f], f"truth $s$ ({f})", dict(norm="hist", min=lo, max=hi)),
                  (d[f], f"observation $d=s+n$ ({f})", dict(norm="hist", min=lo, max=hi)),
                  (u[f], f"denoised $u$ ({f})", dict(norm="hist", min=lo, max=hi)),
                  (err[f], f"error $u-s$ ({f})", dict(min=-v, max=v)),
                  (noise[f], f"noise $d-s$ ({f})", dict(min=-v, max=v))]
        for col, (m, title, kw) in enumerate(panels):
            hp.mollview(m, nest=True, fig=fig.number, sub=(2, 5, 5 * row + col + 1), title=title, cmap="RdBu_r",
                        notext=True, unit=r"$\mu$K", **kw)
    fig.savefig(os.path.join(a.out, "sim_maps.pdf"), bbox_inches="tight")
    plt.close(fig)

    # EE and BB over |b| > 30 and 60 deg.
    theta, _ = hp.pix2ang(n, np.arange(12 * n**2))
    lat = np.abs(90.0 - np.degrees(theta))
    edges = np.unique(np.round(np.geomspace(2, a.lmax + 1, 30)).astype(int))
    ell = 0.5 * (edges[:-1] + edges[1:] - 1)
    others = sorted(glob.glob(a.apply)) if a.apply else []

    def binned(cl):
        dl = np.arange(cl.size) * (np.arange(cl.size) + 1) * cl / (2 * np.pi)
        return np.array([dl[i:j].mean() for i, j in zip(edges[:-1], edges[1:])])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), sharex=True)
    for row, cut in enumerate((30, 60)):
        mask = np.clip(hp.smoothing((lat > cut).astype(float), fwhm=np.radians(a.apod)), 0, 1)
        w2 = np.mean(mask**2)

        def spec(m):
            maps = [np.zeros(12 * n**2)] + [hp.reorder(m[f], n2r=True) * mask for f in qu]
            cl = hp.anafast(maps, lmax=a.lmax, iter=3) / w2
            return cl[1], cl[2]

        curves = [("truth $s$", spec(s), dict(color="k", lw=1.5)),
                  ("observation $d$", spec(d), dict(color="0.4", lw=1.2)),
                  ("denoised $u$", spec(u), dict(color="C0", lw=1.5, ls="--")),
                  ("error $u-s$", spec(err), dict(color="C3", lw=1.5)),
                  ("noise $n=d-s$", spec(noise), dict(color="0.7", lw=1.5))]
        if others:
            e = [spec({f: z[f] - s[f] for f in qu}) for z in map(np.load, others)]
            curves.append((f"error $u-s$, {len(e)} other noises", tuple(np.mean([x[i] for x in e], 0) for i in range(2)),
                           dict(color="C1", lw=1, ls=":")))
        for label, (ee, bb), style in curves:
            for ax, cl in zip(axes[row], (ee, bb)):
                ax.loglog(ell, np.abs(binned(cl)), label=label, **style)
        for ax, t in zip(axes[row], ("EE", "BB")):
            ax.set_title(rf"{t}, $|b|>{cut}^\circ$")
        axes[row, 0].set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
    for ax in axes[1]:
        ax.set_xlabel(r"$\ell$")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "sim_spectra.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
