"""Figures and tables of the paper from the `train_denoise.py` runs.

    python make_figures.py RUNS_DIR OUT_DIR

RUNS_DIR holds the sphere-convolution runs sph_base, sph_cosmo, sph_pr3,
sph_pr3x4 and the face-image run n64_base (train.json, eval_{Q,U}.npz).
Writes OUT_DIR/*.pdf and OUT_DIR/results_table.tex.

Edge ratio: rms of the 4-children detail of the error u - s on the pixels
along the base-face edges over the rms on the other pixels (1 when the
error ignores the faces).
"""

import json
import os
import sys

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = [("sph_base", r"uniform, gauge $\phi$"),
        ("sph_cosmo", r"uniform, gauge cosmo"),
        ("sph_pr3", r"PR3 $\sigma(p)$"),
        ("sph_pr3x4", r"PR3 $4\,\sigma(p)$")]
FACES = ("n64_base", r"uniform, face images")
FAMILIES = ("var", "S1", "S2", "S3", "S4")


def edge_mask(nside):
    from stlia import healpix_faces as hf

    m = np.zeros((12, nside, nside), bool)
    m[:, 0], m[:, -1], m[:, :, 0], m[:, :, -1] = True, True, True, True
    out = np.zeros(12 * nside * nside, bool)
    out[hf.face_nest(nside).ravel()] = m.ravel()
    return out


def edge_ratio(maps):
    r = maps["u"] - maps["sky"]
    det = r - np.repeat(r.reshape(-1, 4).mean(1), 4)
    e = edge_mask(int(round(np.sqrt(r.size / 12))))
    return float(np.sqrt((det[e] ** 2).mean() / (det[~e] ** 2).mean()))


def load(runs, name):
    with open(os.path.join(runs, name, "train.json")) as f:
        tj = json.load(f)
    maps = {f: dict(np.load(os.path.join(runs, name, f"eval_{f}.npz"))) for f in ("Q", "U")}
    return tj, maps


def final_eval(tj):
    return tj["evals"][str(max(int(k) for k in tj["evals"]))]


def maps_figure(maps, out, title):
    fig = plt.figure(figsize=(10, 5.2))
    cols = [("sky", r"sky $s$"), ("d", r"$d=s+n$"), ("u", r"$u=f_\theta(d)$"),
            ("dres", r"$d-s$"), ("ures", r"$u-s$")]
    for row, f in enumerate(("Q", "U")):
        m = maps[f]
        m = {**m, "dres": m["d"] - m["sky"], "ures": m["u"] - m["sky"]}
        lim = np.percentile(np.abs(m["sky"]), 99)
        rlim = np.percentile(np.abs(m["dres"]), 99)
        for c, (k, lab) in enumerate(cols):
            v = rlim if k.endswith("res") else lim
            hp.mollview(m[k], nest=True, fig=fig.number, sub=(2, 5, 5 * row + c + 1), min=-v, max=v,
                        cmap="RdBu_r", title=f"{f}: {lab}", cbar=True, unit=r"$\mu$K", notext=True)
    fig.suptitle(title, y=0.99)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    runs, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    data = {n: load(runs, n) for n, _ in RUNS + [FACES]}

    maps_figure(data["sph_base"][1], os.path.join(out, "maps_uniform.pdf"),
                r"Uniform white noise, $\sigma = 0.5\,{\rm std}(Q)$; trained on Q, tested on U")
    maps_figure(data["sph_pr3x4"][1], os.path.join(out, "maps_pr3x4.pdf"),
                r"White noise with $4\times$ the PR3 per-pixel level; trained on Q, tested on U")

    fig = plt.figure(figsize=(9, 4.6))
    cases = [FACES, RUNS[0], RUNS[1]]
    for row, f in enumerate(("Q", "U")):
        lim = np.percentile(np.abs(data[FACES[0]][1][f]["d"] - data[FACES[0]][1][f]["sky"]), 99) / 2
        for c, (n, lab) in enumerate(cases):
            m = data[n][1][f]
            hp.mollview(m["u"] - m["sky"], nest=True, fig=fig.number, sub=(2, 3, 3 * row + c + 1), min=-lim, max=lim,
                        cmap="RdBu_r", title=f"{f}: $u-s$, {lab}\nedge ratio {edge_ratio(m):.2f}", cbar=True,
                        unit=r"$\mu$K", notext=True)
    fig.savefig(os.path.join(out, "artefacts.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.4))
    for n, lab in RUNS:
        h = data[n][0]["loss_history"]
        ax.loglog(h["step"], np.array(h["loss"]) / h["loss"][0], label=lab)
    ax.set_xlabel("step")
    ax.set_ylabel("training loss / initial loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "loss.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    x = np.arange(len(FAMILIES))
    w = 0.8 / (len(RUNS) + 1)
    for ax, f in zip(axes, ("Q", "U")):
        for i, (n, lab) in enumerate(RUNS):
            e = final_eval(data[n][0])[f]
            ax.bar(x + i * w, [e["stats_u_vs_sky"][k] for k in FAMILIES], w, label=f"$u$, {lab}")
            ax.scatter(x + i * w, [e["stats_d_vs_sky"][k] for k in FAMILIES], marker="_", s=120, color="k",
                       zorder=3, label="$d$ (no denoising)" if i == 0 else None)
        ax.set_yscale("log")
        ax.set_xticks(x + w * (len(RUNS) - 1) / 2, FAMILIES)
        ax.set_title(f"{f} ({'training' if f == 'Q' else 'test'} sky)")
        ax.axhline(1, color="gray", lw=0.6, ls=":")
    axes[0].set_ylabel(r"rms of $(\Phi - \Phi_{\rm sky})/\sigma_{\rm erg}$")
    axes[1].legend(fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "stats.pdf"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, f in zip(axes, ("Q", "U")):
        for i, (n, lab) in enumerate(RUNS):
            e = final_eval(data[n][0])[f]
            nside = [64 >> k for k in range(len(e["level_power_u_over_sky"]))]
            line, = ax.semilogy(nside, e["level_power_u_over_sky"], "o-", label=f"$u$, {lab}")
            ax.semilogy(nside, e["level_power_d_over_sky"], "x:", color=line.get_color())
        ax.set_xscale("log", base=2)
        ax.axhline(1, color="gray", lw=0.6)
        ax.set_xlabel(r"$N_{\rm side}$ of the level")
        ax.set_title(f"{f} ({'training' if f == 'Q' else 'test'} sky)")
    axes[0].set_ylabel("detail variance / sky")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "levels.pdf"))
    plt.close(fig)

    rows = []
    for n, lab in RUNS + [FACES]:
        tj = data[n][0]
        rms = tj.get("noise_rms") or {f: tj["sigma"] for f in ("Q", "U")}  # uniform runs: one sigma
        for f in ("Q", "U"):
            e = final_eval(tj)[f]
            e0 = tj["evals"]["0"][f]
            cells = " & ".join(f"{e['stats_u_vs_sky'][k]:.2g} ({e0['stats_d_vs_sky'][k]:.2g})" for k in ("S1", "S2", "S3", "S4"))
            rows.append(f"{lab} & {f} & {rms[f]:.3g} & {e['pixel_rms_error_over_noise']:.2f} & "
                        f"{edge_ratio(data[n][1][f]):.2f} & "
                        f"{e['data_loss']:.2g} ({e0['data_loss']:.2g}) & {cells} \\\\")
    with open(os.path.join(out, "results_table.tex"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
