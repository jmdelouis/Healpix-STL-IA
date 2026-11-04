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
    evol_figure(runs, out)


if __name__ == "__main__":
    main()
