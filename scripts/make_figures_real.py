"""Figures of a `train_denoise.py --real` run: the PR3 maps themselves denoised.

    python make_figures_real.py RUN_DIR DATA_DIR OUT_DIR [--nside 256 --sims 0-9]
        [--gnilc DIR --field-mask bk14_mask_gal_n0512.fits]

RUN_DIR holds eval_{Q,U}.npz (d, u); DATA_DIR the PR3 maps
(planck353hm{1,2}_{Q,U}_n{nside}.npy) and the FFP10 simulations
(planck353_noise_{Q,U}_n{nside}_mc{k}.npy).  Writes OUT_DIR/real_maps.pdf
(full sky), real_bicep.pdf (BICEP2 field), real_actpol.pdf (ACTPol D56 field) and real_spectra.pdf (EE and BB
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

RESO = 6.0  # gnomonic views of the fields: 6 arcmin pixels
# BICEP2: galactic (l, b) of the centre (RA 0h, Dec -57.5 deg), 56 deg wide.  ACTPol D56: equatorial box
# -8 < RA < 41 deg, -7.2 < Dec < 4 deg (Louis et al. 2017), shown in equatorial coordinates, 56 x 20 deg.
VIEWS = {"bicep": dict(rot=(316.0, -58.5), coord=None, xsize=560, ysize=560),
         "actpol": dict(rot=(16.5, -1.6), coord=["G", "C"], xsize=560, ysize=200)}
D56 = (-8.0, 41.0, -7.2, 4.0)


def project(m, view):
    """Gnomonic projection of the NESTED galactic map m on a field of VIEWS."""
    return hp.gnomview(m, nest=True, reso=RESO, return_projected_map=True, no_plot=True, **VIEWS[view])


def extent(view):
    """Degrees from the centre of the view."""
    x, y = (VIEWS[view][k] * RESO / 120 for k in ("xsize", "ysize"))
    return [-x, x, -y, y]


def box_mask(nside, box):
    """1 inside the equatorial box (RA0, RA1, Dec0, Dec1) in degrees, RA in [-180, 180), NESTED galactic."""
    theta, phi = hp.pix2ang(nside, np.arange(12 * nside**2), nest=True)
    th, ph = hp.Rotator(coord=["G", "C"])(theta, phi)
    ra, dec = (np.degrees(ph) + 180) % 360 - 180, 90 - np.degrees(th)
    return ((ra > box[0]) & (ra < box[1]) & (dec > box[2]) & (dec < box[3])).astype(float)


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
    p.add_argument("--lmax", type=int, default=None, help="default 3 nside - 1")
    p.add_argument("--gnilc", help="directory of gnilc353_{Q,U}_n{nside}.npy (scripts/prepare_gnilc.py)")
    p.add_argument("--field-mask", help="BICEP/Keck field mask in galactic coordinates (bk14_mask_gal_n0512.fits)")
    a = p.parse_args()
    a.lmax = a.lmax or 3 * a.nside - 1
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

    # Fields: d, u and (with --gnilc) the GNILC dust map on one linear scale, then d - u and the half-mission
    # difference on the scale of the noise, with the outline of the field (BICEP2 with --field-mask).
    gnilc = {f: np.load(os.path.join(a.gnilc, f"gnilc353_{f}_n{a.nside}.npy")) for f in ev} if a.gnilc else None
    fields = {"actpol": (box_mask(a.nside, D56), [0.5], ["-"])}
    if a.field_mask:
        w = np.nan_to_num(hp.read_map(a.field_mask))  # NaN outside the field
        # outer edge of the field (1% of the peak weight) and half the peak weight
        fields["bicep"] = (hp.ud_grade(w, a.nside, order_out="NESTED"), [0.01, 0.5], ["-", "--"])
    for view in ("bicep", "actpol"):
        weight, levels, styles = fields.get(view, (None, None, None))
        field = None
        if weight is not None:
            field = project(weight, view).filled(0)
            field /= field.max()
            sel = weight > levels[0] * weight.max()
            if gnilc:
                rms = np.std  # about the mean of the field, as the paper quotes them
                for f in ev:
                    g, uf, df = gnilc[f][sel], u[f][sel], d[f][sel]
                    print(f"{view} {f}: corr(u, G) {np.corrcoef(uf, g)[0, 1]:.3f} corr(d, G) "
                          f"{np.corrcoef(df, g)[0, 1]:.3f} | rms u {rms(uf):.1f} G {rms(g):.1f} d {rms(df):.1f} "
                          f"u-G {rms(uf - g):.1f} d-G {rms(df - g):.1f} d-u {rms(df - uf):.1f} uK", flush=True)
        cols = 5 if gnilc else 4
        wide = view == "actpol"  # a long strip: one row per panel, Q and U side by side
        fig, axes = plt.subplots(cols, 2, figsize=(10, 2.1 * cols)) if wide else \
            plt.subplots(2, cols, figsize=(3.3 * cols, 7.0))
        for row, f in enumerate(("Q", "U")):
            dust = project(u[f], view).filled(0)
            inside = field > levels[0] if field is not None else np.ones(dust.shape, bool)
            vs = np.percentile(np.abs(dust[inside]), 99.5)  # the scale of the dust in the field
            vn = np.percentile(np.abs(half[f]), 99)
            panels = [(d[f], f"PR3 map $d$ ({f})", vs), (u[f], f"denoised $u$ ({f})", vs)]
            if gnilc:
                panels.append((gnilc[f], f"GNILC ({f})", vs))
            panels += [(res[f], f"$d-u$ ({f})", vn), (half[f], f"$(d_1-d_2)/2$ ({f})", vn)]
            for ax, (m, title, v) in zip(axes[:, row] if wide else axes[row], panels):
                im = ax.imshow(project(m, view), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, extent=extent(view))
                if field is not None:
                    ax.contour(field, levels=levels, colors="k", linewidths=0.8, linestyles=styles,
                               extent=extent(view))
                ax.set_title(title, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, orientation="vertical" if wide else "horizontal", fraction=0.05,
                             pad=0.02 if wide else 0.03, label=r"$\mu$K")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"real_{view}.pdf"), bbox_inches="tight")
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
