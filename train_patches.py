"""Train the HEALPix U-Net d -> u on random disc patches with the
cross-statistics losses of Delouis et al. (2022, A&A 668, A122, Sect. 2.3).

    python train_patches.py --data DIR --out RUN [--nside 256 --j 4 --radius 10 --core 4
                            --patches 16 --realizations 4 --loss l1=1,l2=1,l3=1 --steps 3000]

Mock data (the clean sky is known, for the evaluation): the sky s is the
Planck PR3 353 GHz map of the field (DIR/planck353_{field}_n{nside}.npy); an
observation drawn with the FFP10 simulation k is the triple
`d = s + n_k`, `d1 = s + n1_k`, `d2 = s + n2_k` (full mission and the two half
missions, DIR/planck353_{noise,noisehm1,noisehm2}_{field}_n{nside}_mc{k}.npy,
the same realisation k for the three, with one random sign); T is the
857 GHz intensity map (DIR/planck857_I_n{nside}.npy).

Every step draws `--patches` disc centres at random among the pixel
centres of `--centre-nside` (cached disc geometries), an
observation (simulation of `--sims-train`), and `--realizations` noise
triples (ñ, ñ1, ñ2) from other training simulations; the network runs on the
discs of radius `--radius` (`stlia.regions.Patches`), and the losses use the
cross statistics S(a, b) of STL-Dev-MPI on the same discs, averaged over the
core of radius `--core` (the ring between them absorbs the edge effects of
the network and of the wavelets; with 3 levels at nside 256 the network on a
disc equals the network on the sky within ~4 deg of the centre):

  l1   S(d1, d2)  vs  < S(u + ñ1, u + ñ2) >    statistics of u (half missions)
  l2   S(d, u)    vs  < S(u + ñ, u) >          alignment of u with the sky
  l3   S(T, d)    vs  < S(T, u + ñ) >          dependence on the 857 GHz map

each a reduced chi^2 over the coefficients, weighted by the ergodic variance
of its target on the disc (`PatchEngine` moments / native pixels), averaged
over the discs.  The noise-induced biases B_k of Delouis et al. are the
differences between the two sides: they are estimated at every step from the
realisations instead of being frozen between batches of iterations.  All the
pairs of a step go through one `forward_cross` / `vjp_cross` of the engine.

The evaluation applies the network to the whole sky (fields `--train-field`
and `--test-field`, test simulation `--sims-test`[0]) and compares u with the
clean sky: pixel error relative to the noise, auto statistics of u and d
against those of the sky (rms per family in ergodic sigma), and the three
losses on fixed discs with test simulations.

Writes RUN/model.pt, RUN/train.json, RUN/eval_{field}.npz (sky, d, u).
"""

import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np
import torch

from stlia.regions import Patches
from stlia.stl_torch import DenoiseLoss, as_real
from stlia.unet import UNet

from stl_port.native import NativeEngine, PatchEngine  # noqa: E402  (STL-Dev-MPI python/stl_ensemble)

TERMS = ("l1", "l2", "l3")


def id_range(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


class CrossStats(torch.autograd.Function):
    """Cross statistics of disc pairs: `a`, `b` flat `[sum n_i]` (pair i on
    the disc of `centres[i]`) -> `[pairs, ncoeff, 2]`; the moments of the last
    call are kept in `box`."""

    @staticmethod
    def forward(ctx, a, b, sizes, centres, run):
        pe, box = run["engine"], run["box"]
        run["generation"] += 1
        g = run["generation"]
        pe.select(centres, run["radius"], run["core"])
        sa = np.split(a.detach().cpu().numpy(), np.cumsum(sizes)[:-1])
        sb = np.split(b.detach().cpu().numpy(), np.cumsum(sizes)[:-1])
        values, var = pe.forward_cross(sa, sb, moments=True, generation=g)
        box["var"], box["npix"] = var, pe.native_pixels.copy()
        ctx.run, ctx.g, ctx.sizes, ctx.device = run, g, sizes, a.device
        return torch.from_numpy(as_real(values).copy()).to(a.device)

    @staticmethod
    def backward(ctx, grad):
        cot = grad.detach().cpu().numpy()
        ga, gb = ctx.run["engine"].vjp_cross(cot[..., 0] + 1j * cot[..., 1], generation=ctx.g)
        ga = torch.from_numpy(np.concatenate(ga)).to(ctx.device)
        gb = torch.from_numpy(np.concatenate(gb)).to(ctx.device)
        return ga, gb, None, None, None


def chi2(pred, target, var, npix):
    """Reduced chi^2 per disc: `pred` `[P, M, nc, 2]` (mean over M), `target` `[P, nc, 2]`."""
    sigma2 = torch.as_tensor(np.maximum(var / npix, 1e-300), device=target.device)
    w = torch.where(torch.as_tensor(var > 0, device=target.device), 1.0 / sigma2, torch.zeros_like(sigma2))
    w = torch.where(torch.as_tensor(var > 0, device=target.device), w, w.max(1, keepdim=True).values)
    return (w * ((pred.mean(1) - target) ** 2).sum(-1)).mean(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nside", type=int, default=256)
    p.add_argument("--j", type=int, default=4, help="scales of the statistics on a disc")
    p.add_argument("--radius", type=float, default=10.0, help="disc radius [deg]")
    p.add_argument("--core", type=float, default=4.0, help="averaging radius of the statistics [deg]")
    p.add_argument("--patches", type=int, default=16)
    p.add_argument("--centre-nside", type=int, default=16,
                   help="disc centres drawn among the pixel centres of this nside (0: every pixel); "
                        "a finite set lets the engine reuse its cached disc geometries")
    p.add_argument("--geometry-cache", type=int, default=4096)
    p.add_argument("--realizations", type=int, default=4)
    p.add_argument("--train-field", default="Q")
    p.add_argument("--test-field", default="U")
    p.add_argument("--sims-train", default="10-49")
    p.add_argument("--sims-test", default="0-9")
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--loss", default="l1=1,l2=1,l3=1")
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--channels", default="32,64,96")
    p.add_argument("--gauge", default="phi", choices=["phi", "cosmo"])
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=250)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    weights = {t: 0.0 for t in TERMS}
    for item in a.loss.split(","):
        k, _, v = item.partition("=")
        assert k in TERMS, f"unknown loss term {k}"
        weights[k] = float(v or 1)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    npix = 12 * a.nside**2

    def load(name, k=None):
        tag = f"_mc{k}" if k is not None else ""
        return torch.as_tensor(np.load(os.path.join(a.data, f"{name}_n{a.nside}{tag}.npy")), device=dev)

    fields = (a.train_field, a.test_field)
    sky = {f: load(f"planck353_{f}") for f in fields}
    T = load("planck857_I")
    train_ids, test_ids = id_range(a.sims_train), id_range(a.sims_test)

    def noise(f, k):
        """(n, n1, n2) of simulation k for field f, times noise_scale."""
        return tuple(a.noise_scale * load(f"planck353_{kind}_{f}", k) for kind in ("noise", "noisehm1", "noisehm2"))

    cache = {}

    def noise_c(f, k):
        if (f, k) not in cache:
            cache[(f, k)] = noise(f, k)
        return cache[(f, k)]

    pe = PatchEngine(a.nside, a.j, cross=True, threads=a.threads, cache=a.geometry_cache)
    # Candidate centres: the pixel of `nside` that holds each pixel centre of `centre_nside`.
    if a.centre_nside:
        from stlia.regions import pixel_vectors
        fine, coarse = pixel_vectors(a.nside, "cpu"), pixel_vectors(a.centre_nside, "cpu")
        candidates = torch.argmax(coarse @ fine.T, 1).numpy() if a.nside <= 128 else np.array(
            [int(torch.argmax(fine @ v)) for v in coarse])
    else:
        candidates = np.arange(npix)
    run = {"engine": pe, "box": {}, "generation": 0, "radius": a.radius, "core": a.core}
    ncoeff = pe.ncoeff
    channels = [int(c) for c in a.channels.split(",")]
    scale = float(sky[a.train_field].std())
    net = UNet(a.nside, a.levels, channels, gauge=a.gauge, norm="pixel", scale=scale).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    nparams = sum(q.numel() for q in net.parameters())
    M = a.realizations

    def losses(f, centres, k, ks, signs, grad=True):
        """The three terms on the discs of `centres` for the observation of
        simulation k and the realisations ks (field f)."""
        ids_np = pe.select(centres, a.radius, a.core)
        ids = [torch.as_tensor(i, device=dev) for i in ids_np]
        pt = Patches(ids, a.nside, a.levels, gauge=a.gauge)
        n, n1, n2 = noise_c(f, k)
        s = sky[f]
        d, d1, d2 = s + signs[0] * n, s + signs[0] * n1, s + signs[0] * n2
        u = net.forward_patches(torch.cat([d[i] for i in ids])[None], pt)[0]
        us = pt.split(u)
        real = [noise_c(f, kk) for kk in ks]
        A, Bm, C = [], [], []  # pairs (a, b) and their centres
        # targets: S(d1, d2), S(d, u), S(T, d)
        for i, c in enumerate(centres):
            A += [d1[ids[i]], d[ids[i]], T[ids[i]]]
            Bm += [d2[ids[i]], us[i], d[ids[i]]]
            C += [c, c, c]
        # realisations: S(u + ñ1, u + ñ2), S(u + ñ, u), S(T, u + ñ)
        for i, c in enumerate(centres):
            for m, (rn, rn1, rn2) in enumerate(real):
                sg = signs[1 + m]
                A += [us[i] + sg * rn1[ids[i]], us[i] + sg * rn[ids[i]], T[ids[i]]]
                Bm += [us[i] + sg * rn2[ids[i]], us[i], us[i] + sg * rn[ids[i]]]
                C += [c, c, c]
        sizes = [len(x) for x in A]
        st = CrossStats.apply(torch.cat(A), torch.cat(Bm), sizes, np.asarray(C), run)
        var, npx = run["box"]["var"], run["box"]["npix"]
        P = len(centres)
        tgt, tvar, tnp = st[:3 * P].view(P, 3, ncoeff, 2), var[:3 * P].reshape(P, 3, ncoeff), npx[:3 * P].reshape(P, 3, ncoeff)
        pred = st[3 * P:].view(P, len(real), 3, ncoeff, 2)
        out = {}
        for t, name in enumerate(TERMS):
            target = tgt[:, t] if name == "l2" else tgt[:, t].detach()
            out[name] = chi2(pred[:, :, t], target, tvar[:, t], tnp[:, t]).mean()
        return out, d, u

    def draw(pool):
        k = int(rng.choice(pool))
        ks = [int(x) for x in rng.choice([x for x in pool if x != k], M, replace=False)]
        return k, ks, rng.choice([-1.0, 1.0], M + 1)

    full = NativeEngine(a.nside, int(math.log2(a.nside)) - 1, batch=1, backend="cpu")
    rep = DenoiseLoss(full.plan, a.nside, device=dev)

    def auto(m):
        v, var = full.forward(m.detach().cpu().numpy()[None], moments=True)
        return torch.as_tensor(as_real(v[0]), device=dev), torch.as_tensor(var[0], device=dev)

    eval_centres = rng.choice(candidates, a.patches, replace=False)

    def evaluate(save=False):
        net.eval()
        out = {}
        with torch.no_grad():
            for f in fields:
                k = test_ids[0]
                n, _, _ = noise_c(f, k)
                s, d = sky[f], sky[f] + n
                u = net(d[None])[0]
                ts, tv = auto(s)
                us, _ = auto(u)
                ds, _ = auto(d)
                ks = [x for x in test_ids if x != k][:M]
                terms, _, _ = losses(f, eval_centres, k, ks, np.ones(M + 1), grad=False)
                out[f] = {
                    "pixel_rms_error_over_noise": float((u - s).square().mean().sqrt() / n.square().mean().sqrt()),
                    "loss_terms": {t: v.item() for t, v in terms.items()},
                    "stats_u_vs_sky": rep.report(us, ts, tv),
                    "stats_d_vs_sky": rep.report(ds, ts, tv),
                }
                if save:
                    np.savez(os.path.join(a.out, f"eval_{f}.npz"), sky=s.cpu().numpy(), d=d.cpu().numpy(), u=u.cpu().numpy())
        net.train()
        return out

    def show(step, ev):
        for f, e in ev.items():
            fam = " ".join(f"{k} {v:.3g}/{e['stats_d_vs_sky'][k]:.3g}" for k, v in e["stats_u_vs_sky"].items())
            terms = " ".join(f"{k} {v:.4g}" for k, v in e["loss_terms"].items())
            print(f"eval {step} {f}: pixel err/noise {e['pixel_rms_error_over_noise']:.3f} {terms} "
                  f"| u/d vs sky [sigma]: {fam}", flush=True)

    print(f"nside {a.nside} disc J {a.j} R {a.radius} core {a.core} coefficients {ncoeff} levels {a.levels} "
          f"channels {channels} params {nparams} patches {a.patches} realisations {M} loss {weights} "
          f"device {dev} threads {torch.get_num_threads()} scale {scale:.4g}", flush=True)
    hist = {"step": [], "loss": [], **{t: [] for t in TERMS}}
    evals = {0: evaluate()}
    show(0, evals[0])
    started = time.time()
    for step in range(1, a.steps + 1):
        centres = rng.choice(candidates, a.patches, replace=False)
        k, ks, signs = draw(train_ids)
        terms, _, _ = losses(a.train_field, centres, k, ks, signs)
        loss = sum(weights[t] * v for t, v in terms.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == 1:
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            for t in TERMS:
                hist[t].append(terms[t].item())
            print(f"step {step} loss {loss.item():.6g} " + " ".join(f"{t} {v.item():.4g}" for t, v in terms.items())
                  + f" {(time.time() - started) / step:.3f} s/step"
                  + f" maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GiB", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evals[step] = evaluate(save=step == a.steps)
            show(step, evals[step])
            torch.save({"state": net.state_dict(), **net.config(), "j": a.j}, os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "train.json"), "w") as fh:
        json.dump({"args": vars(a), "parameters": nparams, "coefficients": ncoeff, "loss_weights": weights,
                   "seconds": time.time() - started, "loss_history": hist, "evals": evals}, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
