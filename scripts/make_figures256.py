"""Figures and table of the nside 256 runs (FFP10 noise draws).

    python make_figures256.py RUNS_DIR OUT_DIR

RUNS_DIR holds n256_data (full sky, auto loss), n256_delouis (full sky, the
three cross losses of Delouis et al. 2022) and p256_l123 (disc patches, same
losses), each with train.json and eval_{Q,U}.npz.  Writes OUT_DIR/maps256.pdf,
OUT_DIR/evol256.pdf and OUT_DIR/results256_table.tex.
"""

import json
import os
import re
import sys

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = [("n256_data", "full sky, auto"),
        ("n256_delouis", r"full sky, $L_1{+}L_2{+}L_3$"),
        ("p256_l123", r"patches, $L_1{+}L_2{+}L_3$")]
FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")


def evals_from_log(path):
    """Evaluations of a run still in progress, from the `eval` lines of its log."""
    evals = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"eval (\d+) (\w): pixel err/noise ([\d.]+)", line)
            if not m:
                continue
            pairs = re.findall(r"\b(mean|var|S\d) ([-\d.e+]+)/([-\d.e+]+)", line.split("|", 1)[1])
            evals.setdefault(m.group(1), {})[m.group(2)] = {
                "pixel_rms_error_over_noise": float(m.group(3)),
                "stats_u_vs_sky": {k: float(u) for k, u, _ in pairs},
                "stats_d_vs_sky": {k: float(d) for k, _, d in pairs}}
    return {"evals": evals, "partial": True}


def load(runs, name):
    """(train.json, eval maps); a run without train.json yet: evals from its log, no maps."""
    path = os.path.join(runs, name, "train.json")
    if not os.path.exists(path):
        return evals_from_log(os.path.join(runs, name, "log")), None
    with open(path) as f:
        tj = json.load(f)
    maps = {f: dict(np.load(os.path.join(runs, name, f"eval_{f}.npz"))) for f in ("Q", "U")}
    return tj, maps


def final_eval(tj):
    return tj["evals"][str(max(int(k) for k in tj["evals"]))]


def fmt(x):
    return f"{x:.2g}" if x < 10 else f"{x:.0f}"


def table(runs, out):
    rows = []
    for name, label in RUNS:
        tj, maps = runs[name]
        ev = final_eval(tj)
        if tj.get("partial"):
            label += rf"$^\dagger$ ({max(int(k) for k in tj['evals'])})"
        for f in ("Q", "U"):
            e = ev[f]
            cells = [label, f, f"{e['pixel_rms_error_over_noise']:.2f}"]
            cells += [f"{fmt(e['stats_u_vs_sky'][k])} ({fmt(e['stats_d_vs_sky'][k])})" for k in FAMILIES]
            rows.append(" & ".join(cells) + r" \\")
    with open(os.path.join(out, "results256_table.tex"), "w") as fh:
        fh.write("\n".join(rows) + "\n")


def maps_figure(runs, out):
    done = [(n, lab) for n, lab in RUNS if runs[n][1] is not None]
    fig = plt.figure(figsize=(16, 5.2))
    ncol = 2 + len(done)
    for row, f in enumerate(("Q", "U")):
        m0 = runs[done[0][0]][1][f]
        sky = m0["sky"]
        noise = m0["d"] - sky
        lim = 0.5 * np.percentile(np.abs(noise), 99)
        slim = np.percentile(np.abs(sky), 99)
        panels = [(sky, "sky " + f, slim), (noise, "noise $d-s$", lim)]
        panels += [(runs[n][1][f]["u"] - runs[n][1][f]["sky"], "$u-s$: " + lab, lim) for n, lab in done]
        for col, (m, title, v) in enumerate(panels):
            hp.mollview(m, nest=True, fig=fig.number, sub=(2, ncol, row * ncol + col + 1), title=title,
                        min=-v, max=v, cmap="RdBu_r", cbar=True, notext=True)
    fig.savefig(os.path.join(out, "maps256.pdf"), bbox_inches="tight")
    plt.close(fig)


BICEP2 = (316.0, -58.5)  # galactic (l, b) of the centre of the BICEP2 field (RA 0h, Dec -57.5 deg)


def denoise_real(runs_dir, name, maps):
    """Output of the run's network on the PR3 map itself (the real data, d = s),
    cached in RUNS_DIR/name/real_{Q,U}.npy."""
    out = {}
    for f in ("Q", "U"):
        path = os.path.join(runs_dir, name, f"real_{f}.npy")
        if not os.path.exists(path):
            import torch

            from stlia.unet import load as load_net

            net, ck = load_net(os.path.join(runs_dir, name, "model.pt"), "cpu")
            x = torch.as_tensor(maps[f]["sky"], dtype=torch.float32)
            with torch.no_grad():
                u = net(x[None], ck.get("scale") or x.std())[0]
            np.save(path, u.numpy())
        out[f] = np.load(path)
    return out


def bicep_figure(runs, out, runs_dir, reso=6.0, xsize=320, real="n256_data"):
    """Gnomonic views (32 deg) of the BICEP2 field: sky, observation, outputs, and
    the output of the `real` run on the PR3 map itself; histogram-equalised colours."""
    done = [(n, lab) for n, lab in RUNS if runs[n][1] is not None]
    ureal = denoise_real(runs_dir, real, runs[real][1])
    ncol = 3 + len(done)
    fig = plt.figure(figsize=(3.2 * ncol, 7.2))
    for row, f in enumerate(("Q", "U")):
        m0 = runs[done[0][0]][1][f]
        panels = [(m0["sky"], f"PR3 map $s$ ({f})"), (m0["d"], "mock observation $d=s+n$")]
        panels += [(runs[n][1][f]["u"], "$u(d)$: " + lab) for n, lab in done]
        panels += [(ureal[f], "$u(s)$: PR3 map denoised")]
        for col, (m, title) in enumerate(panels):
            hp.gnomview(m, nest=True, rot=BICEP2, reso=reso, xsize=xsize, fig=fig.number,
                        sub=(2, ncol, row * ncol + col + 1), title=title, norm="hist", cmap="RdBu_r",
                        notext=True, unit=r"$\mu$K")
    fig.savefig(os.path.join(out, "bicep256.pdf"), bbox_inches="tight")
    plt.close(fig)


def spectra_figure(runs, out, runs_dir, bcut=30.0, apod=5.0, lmax=600, real="n256_data"):
    """EE and BB pseudo-spectra over |b| > bcut of d, of the noise n = d - s, of the
    sky s and of every u.  The training field Q and the test field U are put
    together as one polarisation map; the mask is smoothed by `apod` deg FWHM."""
    done = [(n, lab) for n, lab in RUNS if runs[n][1] is not None]
    m0 = runs[done[0][0]][1]
    nside = int(round(np.sqrt(m0["Q"]["sky"].size / 12)))
    theta, _ = hp.pix2ang(nside, np.arange(12 * nside * nside))
    mask = (np.abs(90.0 - np.degrees(theta)) > bcut).astype(float)
    mask = np.clip(hp.smoothing(mask, fwhm=np.radians(apod)), 0, 1)
    w2 = np.mean(mask**2)

    def spec(q, u):
        qu = [hp.reorder(x, n2r=True) * mask for x in (q, u)]
        cl = hp.anafast([np.zeros_like(qu[0]), *qu], lmax=lmax) / w2
        return cl[1], cl[2]

    curves = [("$d$", m0["Q"]["d"], m0["U"]["d"], dict(color="0.4", lw=1.5)),
              ("$n=d-s$", m0["Q"]["d"] - m0["Q"]["sky"], m0["U"]["d"] - m0["U"]["sky"], dict(color="0.7", lw=1.5)),
              ("$s$", m0["Q"]["sky"], m0["U"]["sky"], dict(color="k", ls="--", lw=1))]
    curves += [("$u$: " + lab, runs[n][1]["Q"]["u"], runs[n][1]["U"]["u"], dict(lw=1.2)) for n, lab in done]
    ureal = denoise_real(runs_dir, real, runs[real][1])
    curves += [("$u(s)$: PR3 map denoised", ureal["Q"], ureal["U"], dict(color="m", ls="-.", lw=1.2))]
    edges = np.unique(np.round(np.geomspace(2, lmax + 1, 30)).astype(int))
    ell = 0.5 * (edges[:-1] + edges[1:] - 1)

    def binned(cl):
        dl = np.arange(cl.size) * (np.arange(cl.size) + 1) * cl / (2 * np.pi)
        return np.array([dl[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for label, q, u, style in curves:
        ee, bb = spec(q, u)
        axes[0].loglog(ell, binned(ee), label=label, **style)
        axes[1].loglog(ell, binned(bb), label=label, **style)
    for ax, t in zip(axes, ("EE", "BB")):
        ax.set_title(rf"{t}, $|b|>{bcut:.0f}^\circ$")
        ax.set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$D_\ell=\ell(\ell+1)C_\ell/2\pi$ [$\mu$K$^2$]")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "spectra256.pdf"))
    plt.close(fig)


def evol_figure(runs, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for row, f in enumerate(("Q", "U")):
        for col, key in enumerate(("pixel", "mean", "S1")):
            ax = axes[row, col]
            for name, label in RUNS:
                ev = runs[name][0]["evals"]
                steps = sorted(int(k) for k in ev)
                if key == "pixel":
                    y = [ev[str(s)][f]["pixel_rms_error_over_noise"] for s in steps]
                else:
                    y = [ev[str(s)][f]["stats_u_vs_sky"][key] for s in steps]
                ax.plot(steps, y, "o-", ms=3, label=label)
            if key != "pixel":
                ax.axhline(ev["0"][f]["stats_d_vs_sky"][key], color="k", ls=":", lw=1)
                ax.set_yscale("log")
            else:
                ax.axhline(1, color="k", ls=":", lw=1)
            ax.set_title(f"{f}: " + {"pixel": "pixel error / noise", "mean": r"mean [$\sigma$]",
                                     "S1": r"$S_1$ [$\sigma$]"}[key])
            if row == 1:
                ax.set_xlabel("step")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "evol256.pdf"))
    plt.close(fig)


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    runs = {name: load(src, name) for name, _ in RUNS}
    table(runs, out)
    maps_figure(runs, out)
    bicep_figure(runs, out, src)
    spectra_figure(runs, out, src)
    evol_figure(runs, out)


if __name__ == "__main__":
    main()
