"""Figures of a `train_denoise.py --real` run: the PR3 maps themselves denoised.

    python make_figures_real.py RUN_DIR DATA_DIR OUT_DIR [--nside 256 --sims 0-9]
        [--gnilc DIR --field-mask bk14_mask_gal_n0512.fits]

RUN_DIR holds eval_{Q,U}.npz (d, u); DATA_DIR the PR3 maps
(planck353hm{1,2}_{Q,U}_n{nside}.npy) and the FFP10 simulations
(planck353_noise_{Q,U}_n{nside}_mc{k}.npy).  Writes OUT_DIR/real_maps.pdf
(full sky), real_bicep.pdf (BICEP2 field) and real_spectra.pdf (EE and BB
over |b| > 30 deg: d, u, the half-mission cross spectrum d1 x d2, which has no
noise bias, the residual d - u, the half-mission difference (d1 - d2)/2 and
the FFP10 noise).
"""

import argparse
import os

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BICEP2 = (316.0, -58.5)  # galactic (l, b) of the centre of the BICEP2 field (RA 0h, Dec -57.5 deg)
RESO, XSIZE = 6.0, 560  # gnomonic view of the field: 6 arcmin pixels, 56 deg wide
EXTENT = [-XSIZE * RESO / 120, XSIZE * RESO / 120] * 2  # degrees from the centre


def project(m):
    """Gnomonic projection of the NESTED map m on the BICEP2 field."""
    return hp.gnomview(m, nest=True, rot=BICEP2, reso=RESO, xsize=XSIZE, return_projected_map=True, no_plot=True)


def id_range(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("data")
    p.add_argument("out")
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--sims", default="0-9")
    p.add_argument("--bcut", type=float, default=30.0)
    p.add_argument("--apod", type=float, default=5.0)
    p.add_argument("--lmax", type=int, default=600)
    p.add_argument("--gnilc", help="directory of gnilc353_{Q,U}_n{nside}.npy (scripts/prepare_gnilc.py)")
    p.add_argument("--field-mask", help="BICEP/Keck field mask in galactic coordinates (bk14_mask_gal_n0512.fits)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = {f: dict(np.load(os.path.join(a.run, f"eval_{f}.npz"))) for f in ("Q", "U")}
    d = {f: ev[f]["d"] for f in ev}
    u = {f: ev[f]["u"] for f in ev}
    hm = {h: {f: np.load(os.path.join(a.data, f"planck353hm{h}_{f}_n{a.nside}.npy")) for f in ev} for h in (1, 2)}
    half = {f: 0.5 * (hm[1][f] - hm[2][f]) for f in ev}
    res = {f: d[f] - u[f] for f in ev}

    # Full sky: the data d, the denoised u (same histogram-equalised colours) and the residual d - u.
    fig = plt.figure(figsize=(18, 7.4))
    for row, f in enumerate(("Q", "U")):
        hp.mollview(d[f], nest=True, fig=fig.number, sub=(2, 3, 3 * row + 1), title=f"PR3 map $d$ ({f})", norm="hist",
                    cmap="RdBu_r", notext=True, unit=r"$\mu$K")
        hp.mollview(u[f], nest=True, fig=fig.number, sub=(2, 3, 3 * row + 2), title=f"denoised $u$ ({f})", norm="hist",
                    cmap="RdBu_r", notext=True, unit=r"$\mu$K")
        v = np.percentile(np.abs(half[f]), 99)
        hp.mollview(res[f], nest=True, fig=fig.number, sub=(2, 3, 3 * row + 3), title=f"$d-u$ ({f})",
                    min=-v, max=v, cmap="RdBu_r", notext=True, unit=r"$\mu$K")
    fig.savefig(os.path.join(a.out, "real_maps.pdf"), bbox_inches="tight")
    plt.close(fig)

    # BICEP2 field: d, u and (with --gnilc) the GNILC dust map on one linear scale, then d - u and the
    # half-mission difference on the scale of the noise; the outline of the field with --field-mask.
    gnilc = {f: np.load(os.path.join(a.gnilc, f"gnilc353_{f}_n{a.nside}.npy")) for f in ev} if a.gnilc else None
    field = None
    if a.field_mask:
        w = np.nan_to_num(hp.read_map(a.field_mask))  # NaN outside the field
        field = project(hp.ud_grade(w, a.nside, order_out="NESTED")).filled(0)
        field /= field.max()
    cols = 5 if gnilc else 4
    fig, axes = plt.subplots(2, cols, figsize=(3.3 * cols, 7.0))
    for row, f in enumerate(("Q", "U")):
        dust = project(u[f]).filled(0)
        inside = field > 0.01 if field is not None else np.ones(dust.shape, bool)
        vs = np.percentile(np.abs(dust[inside]), 99.5)  # the scale of the dust in the field
        vn = np.percentile(np.abs(half[f]), 99)
        panels = [(d[f], f"PR3 map $d$ ({f})", vs), (u[f], f"denoised $u$ ({f})", vs)]
        if gnilc:
            panels.append((gnilc[f], f"GNILC ({f})", vs))
        panels += [(res[f], "$d-u$", vn), (half[f], "$(d_1-d_2)/2$", vn)]
        for ax, (m, title, v) in zip(axes[row], panels):
            im = ax.imshow(project(m), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=EXTENT)
            if field is not None:
                # outer edge of the field (1% of the peak weight) and half the peak weight
                ax.contour(field, levels=[0.01, 0.5], colors="k", linewidths=0.8, linestyles=["-", "--"],
                           extent=EXTENT)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, label=r"$\mu$K")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "real_bicep.pdf"), bbox_inches="tight")
    plt.close(fig)

    # EE and BB over |b| > bcut.
    theta, _ = hp.pix2ang(a.nside, np.arange(12 * a.nside**2))
    mask = (np.abs(90.0 - np.degrees(theta)) > a.bcut).astype(float)
    mask = np.clip(hp.smoothing(mask, fwhm=np.radians(a.apod)), 0, 1)
    w2 = np.mean(mask**2)

    def pol(m):
        return [np.zeros(m["Q"].size)] + [hp.reorder(m[f], n2r=True) * mask for f in ("Q", "U")]

    def spec(m1, m2=None):
        cl = hp.anafast(pol(m1), None if m2 is None else pol(m2), lmax=a.lmax) / w2
        return cl[1], cl[2]

    noise = [spec({f: np.load(os.path.join(a.data, f"planck353_noise_{f}_n{a.nside}_mc{k}.npy")) for f in ev})
             for k in id_range(a.sims)]
    noise = tuple(np.mean([n[i] for n in noise], 0) for i in range(2))
    curves = [("$d$", spec(d), dict(color="0.4", lw=1.5)),
              (r"$d_1\times d_2$ (no noise bias)", spec(hm[1], hm[2]), dict(color="k", ls="--", lw=1.2)),
              ("$u$", spec(u), dict(color="C0", lw=1.5)),
              ("$d-u$", spec(res), dict(color="C3", lw=1.2)),
              ("$(d_1-d_2)/2$", spec(half), dict(color="C1", lw=1, ls=":")),
              ("FFP10 noise", noise, dict(color="0.7", lw=1.5))]
    edges = np.unique(np.round(np.geomspace(2, a.lmax + 1, 30)).astype(int))
    ell = 0.5 * (edges[:-1] + edges[1:] - 1)

    def binned(cl):
        dl = np.arange(cl.size) * (np.arange(cl.size) + 1) * cl / (2 * np.pi)
        return np.array([dl[i:j].mean() for i, j in zip(edges[:-1], edges[1:])])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for label, (ee, bb), style in curves:
        for ax, cl in zip(axes, (ee, bb)):
            ax.loglog(ell, np.abs(binned(cl)), label=label, **style)
    for ax, t in zip(axes, ("EE", "BB")):
        ax.set_title(rf"{t}, $|b|>{a.bcut:.0f}^\circ$")
        ax.set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "real_spectra.pdf"))
    plt.close(fig)
    for label, (ee, bb), _ in curves:
        print(label, " ".join(f"l{i}-{j}: EE {ee[i:j].mean():.4g} BB {bb[i:j].mean():.4g}"
                              for i, j in ((30, 60), (100, 200), (200, 300), (300, 450))))


if __name__ == "__main__":
    main()
