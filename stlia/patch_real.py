"""Denoising of the PR3 maps by a U-Net trained on disc patches
(``train_patches.py --real``): the loss of :mod:`stlia.train_denoise` on the
real data, computed on discs instead of the full sky.

The observations are the PR3 maps: `d` the full-mission map and `d1`, `d2`
the half-mission maps of Q and U (DIR/planck353_{Q,U}_n{nside}.npy,
DIR/planck353hm{1,2}_{Q,U}_n{nside}.npy); the FFP10 simulations of
`--sims-train` provide the noise realisations ñ, ñ1, ñ2 of the model side.
Every step draws `--patches` disc centres among the candidates (pixel centres
of `--centre-nside`; 2 gives the 48 cells of nside 2), runs the network on the
discs of radius `--radius`, and computes on every disc, for every field, the
terms of `--loss`, averaged over the core of radius `--core`:

  data  S(d, d)    vs  < S(u + ñ, u + ñ) >
  l1    S(d1, d2)  vs  < S(u + ñ1, u + ñ2) >
  l2    S(d, u)    vs  < S(u + ñ, u) >
  l3    S(T, d)    vs  < S(T, u + ñ) >
  cl    C(d1, d2)  vs  C(u, u)           binned pseudo-spectra on the disc

with S the cross statistics of the disc engine (`PatchEngine`) and the
reduced chi^2 of :func:`stlia.train_patches.chi2`.  The terms are summed
over the fields and averaged over the discs.

Fields: `--pol-stats qu`, the Q and U maps; `--pol-stats eb` (with
``--joint``), the E and B maps of the disc: (u_Q, u_U) times a taper equal to
1 in the core and falling to 0 at the edge of the disc (cosine), transformed
to (E, B) by spherical harmonics on the full sky (zero outside the disc), and
read back on the disc.  The spectra of `cl` are those of the same tapered
maps (scalar spectra of Q and U, or EE and BB), divided by the mean square of
the taper; their variance is the Gaussian one of the half-mission cross
spectrum, (C11 C22 + C12^2) / modes, with the modes of the tapered disc.

``--joint``: one network for (Q, U) (`maps=2`); otherwise one network for
both fields, applied to each.  ``--bias frozen``: the noise biases B of the
data and Delouis terms, <S(u + ñ...)> - S(u...), are estimated per disc with
`--bias-realizations` draws at the current u, and kept until the disc is
visited again more than `--bias-every` steps later; a finite set of centres
lets every disc keep its own (the targets, fixed, are computed once per
disc).  Zero on the coefficients without noise bias, unless `--bias-all`.

The evaluation applies the network to the full sky and reports the rms of
d - u over that of the FFP10 noise; the last one writes
RUN/eval_{Q,U}.npz (sky = d, d, u), the files of the full-sky runs on the
PR3 maps, for the same figures.
"""

import json
import math
import os
import time

import numpy as np
import torch

from stlia.regions import Patches
from stlia.sht import HEALPixSHT
from stlia.spectra import bin_edges
from stlia.train_denoise import unbiased
from stlia.train_patches import CrossStats, chi2, id_range
from stlia.unet import UNet

FIELDS_QU = ("Q", "U")
DELOUIS = ("l1", "l2", "l3")


class DiscSpectra:
    """Binned pseudo-spectra `[..., nb]` from flat alm `[..., n_alm]` (the
    layout of :class:`stlia.sht.HEALPixSHT`), for multipoles in
    [lmin, lmax] by bins of `width`."""

    def __init__(self, sht, lmin, lmax, width, device):
        big = sht.lmax
        ell = np.concatenate([np.arange(m, big + 1) for m in range(big + 1)])
        m = np.concatenate([np.full(big - m + 1, m) for m in range(big + 1)])
        w = np.where(m == 0, 1.0, 2.0) / (2 * ell + 1)
        edges = bin_edges(lmin, lmax, width)
        rows, cols, vals = [], [], []
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            sel = np.flatnonzero((ell >= lo) & (ell < hi))
            rows += [i] * len(sel)
            cols += list(sel)
            vals += list(w[sel] / (hi - lo))
        self.binning = torch.sparse_coo_tensor(np.array([rows, cols]), np.array(vals, dtype=np.float32),
                                               (len(edges) - 1, len(ell)), device=device).coalesce()
        self.modes = np.array([(2 * np.arange(lo, hi) + 1).sum() for lo, hi in zip(edges[:-1], edges[1:])])

    def __call__(self, a1, a2=None):
        p = (a1 * (a1 if a2 is None else a2).conj()).real.float()
        lead = p.shape[:-1]
        out = torch.sparse.mm(self.binning, p.reshape(-1, p.shape[-1]).T).T
        return out.reshape(*lead, -1)


def run_real(a, weights, dev, pe, candidates, rng):
    """Training on the PR3 maps (``--real``): see the module docstring."""
    nside, npix = a.nside, 12 * a.nside ** 2
    eb = a.pol_stats == "eb"
    assert not eb or a.joint, "--pol-stats eb: --joint"
    fields = ("E", "B") if eb else FIELDS_QU
    F = len(fields)

    def load(name, k=None):
        tag = f"_mc{k}" if k is not None else ""
        return torch.as_tensor(np.load(os.path.join(a.data, f"{name}_n{nside}{tag}.npy")), device=dev,
                               dtype=torch.float32)

    d = torch.stack([load(f"planck353_{f}") for f in FIELDS_QU])  # [2, npix]
    hm = [torch.stack([load(f"planck353hm{h}_{f}") for f in FIELDS_QU]) for h in (1, 2)]
    T = load("planck857_I")
    train_ids = id_range(a.sims_train)
    cache = {}

    def noise(k):
        """(n, n1, n2) `[2, npix]` each, of simulation k, for (Q, U)."""
        if k not in cache:
            cache[k] = tuple(torch.stack([load(f"planck353_{kind}_{f}", k) for f in FIELDS_QU])
                             for kind in ("noise", "noisehm1", "noisehm2"))
        return cache[k]

    noise_rms = torch.stack([noise(k)[0] for k in train_ids[:8]]).square().mean((0, 2)).sqrt()  # [2]

    conds = []
    if "log857" in a.cond:
        i857 = T.cpu().numpy()
        conds.append(np.log(np.clip(i857, np.percentile(i857[i857 > 0], 0.1), None)))
    if "sigma" in a.cond:
        var = torch.stack([noise(k)[0] for k in train_ids]).square().mean((0, 1))
        conds.append(0.5 * np.log(var.cpu().numpy()))
    cond = None
    if conds:
        c = np.stack([(x - x.mean()) / x.std() for x in conds])
        cond = torch.as_tensor(c, device=dev, dtype=torch.float32)  # [cond, npix]

    channels = [int(c) for c in a.channels.split(",")]
    scale = float(d.std())
    net = UNet(nside, a.levels, channels, gauge=a.gauge, norm="pixel", scale=scale,
               cond=len(conds), maps=2 if a.joint else 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    nparams = sum(q.numel() for q in net.parameters())
    run = {"engine": pe, "box": {}, "generation": 0, "radius": a.radius, "core": a.core, "gpu": a.stl_backend == "gpu"}
    nc = pe.ncoeff
    exact = {t: torch.as_tensor(unbiased(pe.plan, t), device=dev) for t in ("data",) + DELOUIS}

    sht = HEALPixSHT(nside, dtype=torch.float32, device=dev)
    spec = DiscSpectra(sht, a.cl_lmin, a.cl_lmax or 2 * nside - 1, a.cl_width, dev) if weights["cl"] else None

    # -- disc geometry: pixels, taper, spectra normalisation (per centre, cached)
    geo = {}

    def geometry(c):
        if c not in geo:
            ids = torch.as_tensor(pe.select(np.array([c]), a.radius, a.core)[0], device=dev)
            v = torch.as_tensor(pe.centre_vectors(np.array([c]))[0], device=dev, dtype=torch.float64)
            th, ph = _angles(ids, nside)
            vec = torch.stack([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()], 1)
            r = torch.rad2deg(torch.arccos((vec @ v).clamp(-1, 1)))
            x = ((r - a.core) / (a.radius - a.core)).clamp(0, 1)
            w = (0.5 * (1 + torch.cos(math.pi * x))).float()
            w2 = float((w ** 2).sum()) / npix
            w4 = float((w ** 4).sum()) / npix
            geo[c] = {"ids": ids, "w": w, "w2": w2, "modes": (w2 ** 2 / w4) * spec.modes if spec else None}
        return geo[c]

    def to_fields(X, c, alms=False):
        """(Q, U) values `[k, 2, n]` on the disc of `c` -> the fields `[k, F, n]` of
        the statistics, and with `alms` the tapered alm `[k, F, n_alm]` of the spectra."""
        g = geometry(c)
        if not eb and not alms:
            return X, None
        full = X.new_zeros(X.shape[0], 2, npix)
        full[:, :, g["ids"]] = X * g["w"]
        if eb:
            aE, aB = sht.map2alm_spin(full[:, 0], full[:, 1], spin=2, nest=True)
            al = torch.stack([aE, aB], 1)
            maps = sht.alm2map(al, nest=True)[:, :, g["ids"]]
            return maps, al
        return X, sht.map2alm(full, nest=True)

    # -- fixed targets per disc
    tgt = {}

    def targets(c):
        if c not in tgt:
            g = geometry(c)
            with torch.no_grad():
                X = torch.stack([d[:, g["ids"]], hm[0][:, g["ids"]], hm[1][:, g["ids"]]])
                Fx, al = to_fields(X, c, alms=bool(spec))
                Tc = T[g["ids"]]
                A, Bm = [], []
                for i in range(F):
                    A += [Fx[0, i], Fx[1, i], Tc]  # S(d, d), S(d1, d2), S(T, d)
                    Bm += [Fx[0, i], Fx[2, i], Fx[0, i]]
                st, var, npx = _stats(A, Bm, [c] * len(A), run)
                t = {"maps": Fx, "stats": st.view(F, 3, nc, 2), "var": var.reshape(F, 3, nc),
                     "npix": npx.reshape(F, 3, nc)}
                if spec:
                    c11, c22, c12 = spec(al[1]) / g["w2"], spec(al[2]) / g["w2"], spec(al[1], al[2]) / g["w2"]
                    t["cl"] = c12
                    t["cl_var"] = ((c11 * c22 + c12 ** 2) / torch.as_tensor(g["modes"], device=dev)).clamp_min(1e-300)
                tgt[c] = t
        return tgt[c]

    # -- frozen biases per disc
    bias, bias_step = {}, {}
    bias_gen = np.random.default_rng(a.seed + 777)

    def estimate_bias(c, u):
        """Biases `{term: [F, nc, 2]}` at the (Q, U) values `u` `[2, n]` of the disc of `c`."""
        g = geometry(c)
        Tc = T[g["ids"]]
        with torch.no_grad():
            Fu, _ = to_fields(u[None], c)
            Fu = Fu[0]
            A, Bm = [], []
            for i in range(F):
                A += [Fu[i], Tc]
                Bm += [Fu[i], Fu[i]]
            st0, _, _ = _stats(A, Bm, [c] * len(A), run)
            st0 = st0.view(F, 2, nc, 2)
            acc = {t: 0 for t in ("data",) + DELOUIS}
            M = a.bias_realizations
            for k0 in range(0, M, 4):
                ks = [int(x) for x in bias_gen.choice(train_ids, min(4, M - k0))]
                sg = bias_gen.choice([-1.0, 1.0], len(ks))
                X = torch.cat([torch.stack([s * noise(k)[j][:, g["ids"]] for j in range(3)]) for k, s in zip(ks, sg)])
                Fn, _ = to_fields(X, c)
                Fn = Fn.view(len(ks), 3, F, -1)
                A, Bm = [], []
                for q in range(len(ks)):
                    n, n1, n2 = Fn[q]
                    for i in range(F):
                        A += [Fu[i] + n[i], Fu[i] + n1[i], Fu[i] + n[i], Tc]
                        Bm += [Fu[i] + n[i], Fu[i] + n2[i], Fu[i], Fu[i] + n[i]]
                st, _, _ = _stats(A, Bm, [c] * len(A), run)
                st = st.view(len(ks), F, 4, nc, 2).sum(0)
                for t, name in enumerate(("data", "l1", "l2", "l3")):
                    acc[name] = acc[name] + st[:, t]
            out = {"data": acc["data"] / M - st0[:, 0], "l1": acc["l1"] / M - st0[:, 0],
                   "l2": acc["l2"] / M - st0[:, 0], "l3": acc["l3"] / M - st0[:, 1]}
            if not a.bias_all:
                for t in out:
                    out[t][:, exact[t]] = 0
        return out

    def forward(centres):
        """The network on the discs of `centres`: (Q, U) `[2, n_i]` per disc."""
        ids = [geometry(c)["ids"] for c in centres]
        pt = Patches(ids, nside, a.levels, gauge=a.gauge)
        cd = torch.cat([cond[:, i] for i in ids], -1)[None] if cond is not None else None
        if a.joint:
            u = net.forward_patches(torch.cat([d[:, i] for i in ids], -1)[None], pt, cond=cd)[0]
        else:
            u = torch.stack([net.forward_patches(torch.cat([d[f, i] for i in ids])[None], pt, cond=cd)[0]
                             for f in range(2)])
        return pt.split(u)

    def losses(centres, step, M):
        us = forward(centres)
        live = a.bias == "live" or step <= a.bias_warmup
        if not live:
            for c, u in zip(centres, us):
                if c not in bias or step - bias_step[c] >= a.bias_every:
                    bias[c], bias_step[c] = estimate_bias(c, u.detach()), step
        P = len(centres)
        A, Bm, C = [], [], []
        fu, al = [], []
        for c, u in zip(centres, us):
            t, g = targets(c), geometry(c)
            Fu, au = to_fields(u[None], c, alms=bool(spec))
            fu.append(Fu[0])
            al.append(au)
            Tc = T[g["ids"]]
            if live:  # M noise triples, the same for every field
                ks = [int(x) for x in rng.choice(train_ids, M)]
                sg = rng.choice([-1.0, 1.0], M)
                with torch.no_grad():
                    X = torch.cat([torch.stack([s * noise(k)[j][:, g["ids"]] for j in range(3)])
                                   for k, s in zip(ks, sg)])
                    Fn = to_fields(X, c)[0].view(M, 3, F, -1)
            for i in range(F):
                A += [t["maps"][0, i]]  # S(d, u): the target of l2, with gradient
                Bm += [Fu[0, i]]
                if live:
                    for q in range(M):
                        n, n1, n2 = Fn[q, 0, i], Fn[q, 1, i], Fn[q, 2, i]
                        A += [Fu[0, i] + n, Fu[0, i] + n1, Fu[0, i] + n, Tc]
                        Bm += [Fu[0, i] + n, Fu[0, i] + n2, Fu[0, i], Fu[0, i] + n]
                else:
                    A += [Fu[0, i], Tc]
                    Bm += [Fu[0, i], Fu[0, i]]
            C += [c] * (len(A) - len(C))
        st, var, npx = _stats(A, Bm, C, run, grad=True)
        per = 1 + (4 * M if live else 2)
        st = st.view(P, F, per, nc, 2)
        var, npx = var.reshape(P, F, per, nc), npx.reshape(P, F, per, nc)
        ts = torch.stack([targets(c)["stats"] for c in centres])  # [P, F, 3, nc, 2]
        tv = np.stack([targets(c)["var"] for c in centres])
        tn = np.stack([targets(c)["npix"] for c in centres])
        if live:
            model = st[:, :, 1:].reshape(P, F, M, 4, nc, 2)
            pred = {name: model[:, :, :, t] for t, name in enumerate(("data", "l1", "l2", "l3"))}
        else:
            b = {t: torch.stack([bias[c][t] for c in centres]) for t in ("data",) + DELOUIS}
            pred = {"data": (st[:, :, 1] + b["data"])[:, :, None], "l1": (st[:, :, 1] + b["l1"])[:, :, None],
                    "l2": (st[:, :, 1] + b["l2"])[:, :, None], "l3": (st[:, :, 2] + b["l3"])[:, :, None]}
        target = {"data": (ts[:, :, 0], tv[:, :, 0], tn[:, :, 0]), "l1": (ts[:, :, 1], tv[:, :, 1], tn[:, :, 1]),
                  "l2": (st[:, :, 0], var[:, :, 0], npx[:, :, 0]), "l3": (ts[:, :, 2], tv[:, :, 2], tn[:, :, 2])}
        out = {}
        for name in ("data",) + DELOUIS:
            if not weights[name]:
                continue
            tg, v, n = target[name]
            if name != "l2":
                tg = tg.detach()
            x = chi2(pred[name].reshape(P * F, -1, nc, 2), tg.reshape(P * F, nc, 2),
                     _np(v).reshape(P * F, nc), _np(n).reshape(P * F, nc))
            out[name] = x.view(P, F).sum(1).mean()
        if spec:
            cl = []
            for c, au in zip(centres, al):
                t = targets(c)
                model = spec(au[0]) / geometry(c)["w2"]
                cl.append((((model - t["cl"]) ** 2) / t["cl_var"]).mean(-1).sum())
            out["cl"] = torch.stack(cl).mean()
        return out

    def evaluate(save=False):
        net.eval()
        with torch.no_grad():
            cd = cond[None] if cond is not None else None
            if a.joint:
                u = net(d[None], cond=cd)[0]
            else:
                u = torch.stack([net(d[f][None], cond=cd)[0] for f in range(2)])
        net.train()
        out = {f: {"removed_rms_over_noise": float((d[i] - u[i]).square().mean().sqrt() / noise_rms[i])}
               for i, f in enumerate(FIELDS_QU)}
        if save:
            for i, f in enumerate(FIELDS_QU):
                np.savez(os.path.join(a.out, f"eval_{f}.npz"), sky=d[i].cpu().numpy(), d=d[i].cpu().numpy(),
                         u=u[i].cpu().numpy())
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return out

    def show(step, ev):
        print(f"eval {step}: removed rms / noise " + " ".join(f"{f} {e['removed_rms_over_noise']:.3f}"
                                                              for f, e in ev.items()), flush=True)

    print(f"real PR3 maps, nside {nside} disc J {a.j} R {a.radius} core {a.core} coefficients {nc} "
          f"fields {fields} joint {a.joint} levels {a.levels} channels {channels} params {nparams} "
          f"patches {a.patches} of {len(candidates)} centres, loss {weights}, bias {a.bias}, cond {a.cond}, "
          f"noise rms Q {float(noise_rms[0]):.4g} U {float(noise_rms[1]):.4g}", flush=True)
    terms_all = ("data",) + DELOUIS + ("cl",)
    hist = {"step": [], "loss": [], **{t: [] for t in terms_all}}
    evals = {0: evaluate()}
    show(0, evals[0])
    started = time.time()
    for step in range(1, a.steps + 1):
        centres = [int(c) for c in rng.choice(candidates, a.patches, replace=False)]
        terms = losses(centres, step, a.realizations)
        loss = sum(weights[t] * v for t, v in terms.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == 1:
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            for t in terms_all:
                hist[t].append(terms[t].item() if t in terms else None)
            print(f"step {step} loss {loss.item():.6g} " + " ".join(f"{t} {v.item():.4g}" for t, v in terms.items())
                  + f" {(time.time() - started) / step:.3f} s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evals[step] = evaluate(save=step == a.steps)
            show(step, evals[step])
            path = os.path.join(a.out, "model.pt")
            torch.save({"state": net.state_dict(), **net.config(), "j": a.j}, path + ".tmp")
            os.replace(path + ".tmp", path)
    with open(os.path.join(a.out, "train.json"), "w") as fh:
        json.dump({"args": vars(a), "parameters": nparams, "coefficients": nc, "loss_weights": weights,
                   "seconds": time.time() - started, "loss_history": hist, "evals": evals}, fh, indent=1)


def _np(x):
    return x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()


def _angles(ids, nside):
    """(theta, phi) float64 of the NESTED pixels `ids` (torch)."""
    import healpy as hp
    th, ph = hp.pix2ang(nside, ids.cpu().numpy(), nest=True)
    return (torch.as_tensor(th, device=ids.device), torch.as_tensor(ph, device=ids.device))


def _stats(A, B, centres, run, grad=False):
    """Cross statistics of the pairs (A[i], B[i]) on the discs `centres[i]` ->
    (values `[N, nc, 2]`, variances, pixels) of the engine."""
    sizes = [len(x) for x in A]
    st = CrossStats.apply(torch.cat(A), torch.cat(B), sizes, np.asarray(centres), run)
    return (st if grad else st.detach()), run["box"]["var"], run["box"]["npix"]
