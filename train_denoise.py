"""Train the HEALPix U-Net d -> u with scattering denoising losses.

    python train_denoise.py --data DIR --out RUN [--nside 64 --train-field Q --test-field U
                            --noise pr3sims --realizations 4 --loss data=1,res=1,cross=1
                            --steps 3000 --lr 1e-3 --device 0]

Sky maps: DIR/planck353_{field}_n{nside}.npy (NESTED, uK; `scripts/prepare_planck.py`).

Noise (`--noise`), `noise_scale` times:
  uniform   white, `sigma = noise_rel * std(train sky)`, the same for every field;
  pr3white  white with the per-pixel level of the PR3 simulations,
            DIR/planck353_noise_std_{field}_n{nside}.npy (`scripts/noise_std.py`);
  pr3sims   draws of the degraded FFP10 simulations themselves,
            DIR/planck353_noise_{field}_n{nside}_mc{k}.npy, with a random sign:
            the stripes and correlations of the PR3 noise.  The simulations
            `--sims-train` feed training, `--sims-test` the evaluation only; the
            noise of an observation and its realisations n_i are distinct maps.

Every step draws an observation `d = sky + n_0` of the training sky, `u = f(d)`,
realisations `n_i` (i = 1..B), and minimises a weighted sum of reduced chi^2
on the scattering covariances (`stlia.stl_torch.DenoiseLoss`, ergodic weights):

  data   mean_i S(u + n_i)  vs  S(d)            (u plus noise looks like d)
  res    S(d - u)           vs  mean_i S(n_i)   (what is removed looks like noise)
  cross  S(2u - d)          vs  S(d)            (u - r looks like u + r)

and the full-sky terms of Delouis et al. (2022), with the cross statistics
S(a, b) of the GPU engine and half-mission observations d1 = sky + n1,
d2 = sky + n2 of the same FFP10 simulation as d (T: the 857 GHz map):

  l1   S(d1, d2)  vs  < S(u + ñ1, u + ñ2) >
  l2   S(d, u)    vs  < S(u + ñ, u) >
  l3   S(T, d)    vs  < S(T, u + ñ) >

With r = d - u, d = u + r: if r is a noise independent of u with a symmetric
distribution, u + r and u - r have the same statistics, so `cross` asks the
residual to be uncorrelated with u (for the quadratic terms, by the
polarisation identity: S(u + r) - S(u - r) = 4 S(u, r)), without cross
statistics in the engine; it vanishes for u = d and does not oppose `data`.
(A first form, S(2u - d) vs mean_i S(u - n_i), repeats the mismatch of
`data` with the opposite sign at u = d, and the network stays the identity.)
The weights of `data` and `cross` are the ergodic variances of d, those of
`res` the ergodic variances of a noise map.  No
pixel of the clean sky is used.  The evaluation (fixed noise of the test
simulations, train and test fields) compares `u` with the clean sky: pixel
error relative to the noise, statistics of `u` and of `d` against those of the
sky (rms per family in units of the ergodic sigma of the sky), the variance of
the 4-children details per level, and the three loss terms.

With `--real` the observations are the PR3 maps themselves: d is the
full-mission map and (d1, d2) the half-mission maps of the field, the network
is trained on both fields (train and test fields alternately, the same fixed d
every time: an amortised form of the per-map optimisation of Delouis et al.),
and the noise draws only enter the realisations n_i.  There is no clean sky:
the evaluation reports the rms of d - u over the noise rms (1 when u removes
as much as the noise), and the statistics of u against those of d.

Writes RUN/model.pt, RUN/train.json, RUN/eval_{field}.npz (sky, d, u; with
`--real`, sky = d).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from stlia.stl_torch import DenoiseLoss, Stats, as_real
from stlia.unet import UNet

from stl_port.native import NativeEngine  # noqa: E402  (STL-Dev-MPI python/stl_ensemble)

TERMS = ("data", "res", "cross", "l1", "l2", "l3")
DELOUIS = ("l1", "l2", "l3")


class CrossStats(torch.autograd.Function):
    """Full-sky cross statistics S(a_i, b_i) of the pairs `a`, `b` `[N, npix]`
    (`NativeEngine(..., cross=True)`, GPU) -> `[N, ncoeff, 2]`; the moments of
    the last call are kept in `run["var"]`."""

    @staticmethod
    def forward(ctx, a, b, run):
        run["generation"] += 1
        g = run["generation"]
        values, var = run["engine"].forward_cross(a.detach().cpu().numpy(), b.detach().cpu().numpy(),
                                                  generation=g, moments=True)
        run["var"] = torch.as_tensor(var, device=a.device)
        ctx.run, ctx.g, ctx.device = run, g, a.device
        return torch.from_numpy(as_real(values).copy()).to(a.device)

    @staticmethod
    def backward(ctx, grad):
        cot = grad.detach().cpu().numpy()
        ga, gb = ctx.run["engine"].vjp_cross(cot[..., 0] + 1j * cot[..., 1], generation=ctx.g)
        return torch.from_numpy(np.ascontiguousarray(ga)).to(ctx.device),             torch.from_numpy(np.ascontiguousarray(gb)).to(ctx.device), None


def level_powers(x, levels):
    """`[levels]`: variance of the 4-children details of every level, finest first."""
    out = []
    cur = np.asarray(x, dtype=np.float64)
    for _ in range(levels):
        c = cur.reshape(-1, 4)
        par = c.mean(-1)
        out.append(float(((c - par[:, None]) ** 2).mean()))
        cur = par
    return out


def id_range(text):
    """'10-49' or '0,3,5' -> list of ints."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


class Noise:
    """Noise draws `[k, npix]` for one field."""

    def __init__(self, a, field, sky, load, dev):
        self.mode, self.dev, self.npix = a.noise, dev, sky.numel()
        if a.noise == "uniform":
            self.sigma = torch.full_like(sky, a.noise_scale * a.noise_rel_std)
        elif a.noise == "pr3white":
            self.sigma = a.noise_scale * load(f"noise_std_{field}")
        else:
            kinds = ("noise", "noisehm1", "noisehm2") if a.halfmission else ("noise",)
            self.train = [torch.stack([a.noise_scale * load(f"{kd}_{field}", k) for k in id_range(a.sims_train)])
                          for kd in kinds]
            self.test = [torch.stack([a.noise_scale * load(f"{kd}_{field}", k) for k in id_range(a.sims_test)])
                         for kd in kinds]
        self.rms = float(self.draw(64, test=True).square().mean().sqrt())

    def draw(self, k, test=False, generator=None, hm=False):
        """`[k, npix]` full-mission noise; with `hm`, the triple (n, n1, n2) of
        the same simulations and signs (full and half missions)."""
        if self.mode != "pr3sims":
            assert not hm, "half-mission noise needs --noise pr3sims"
            return self.sigma * torch.randn(k, self.npix, generator=generator, device=self.dev, dtype=self.sigma.dtype)
        pools = self.test if test else self.train
        idx = torch.randperm(len(pools[0]), generator=generator, device=self.dev)[: min(k, len(pools[0]))]
        idx = idx[torch.arange(k, device=self.dev) % len(idx)]  # reuse when k > pool (evaluation of small pools)
        sign = torch.randint(0, 2, (k, 1), generator=generator, device=self.dev, dtype=pools[0].dtype) * 2 - 1
        if hm:
            return tuple(pl[idx] * sign for pl in pools)
        return pools[0][idx] * sign


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--j", type=int, default=None)
    p.add_argument("--train-field", default="Q")
    p.add_argument("--test-field", default="U")
    p.add_argument("--noise", default="pr3sims", choices=["uniform", "pr3white", "pr3sims"])
    p.add_argument("--noise-rel", type=float, default=0.5, help="uniform: sigma / std(train sky)")
    p.add_argument("--noise-scale", type=float, default=1.0, help="factor on every noise model")
    p.add_argument("--sims-train", default="10-49")
    p.add_argument("--sims-test", default="0-9")
    p.add_argument("--realizations", type=int, default=4)
    p.add_argument("--real", action="store_true", help="denoise the PR3 maps themselves (no mock sky)")
    p.add_argument("--loss", default="data=1,res=1,cross=1",
                   help="weights of the terms data, res, cross and of the Delouis et al. (2022) terms l1, l2, l3")
    p.add_argument("--levels", type=int, default=None, help="default: down to nside 8")
    p.add_argument("--channels", default=None, help="comma-separated, one per level")
    p.add_argument("--conv", default="sphere", choices=["sphere", "faces"])
    p.add_argument("--gauge", default="phi", choices=["phi", "cosmo"])
    p.add_argument("--ksz", type=int, default=3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=250)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    j = a.j or int(math.log2(a.nside)) - 1
    levels = a.levels or int(math.log2(a.nside // 8)) + 1
    weights = {t: 0.0 for t in TERMS}
    for item in a.loss.split(","):
        k, _, v = item.partition("=")
        assert k in TERMS, f"unknown loss term {k}"
        weights[k] = float(v or 1)
    dev = torch.device(f"cuda:{a.device}")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    def load(name, k=None):
        tag = f"_mc{k}" if k is not None else ""
        return torch.as_tensor(np.load(os.path.join(a.data, f"planck353_{name}_n{a.nside}{tag}.npy")), device=dev)

    train_sky, test_sky = load(a.train_field), load(a.test_field)
    a.halfmission = any(weights[t] for t in DELOUIS)
    if a.halfmission:
        T = torch.as_tensor(np.load(os.path.join(a.data, f"planck857_I_n{a.nside}.npy")), device=dev)
    a.noise_rel_std = a.noise_rel * float(train_sky.std())
    noise = {f: Noise(a, f, s, load, dev) for f, s in ((a.train_field, train_sky), (a.test_field, test_sky))}
    npix = 12 * a.nside**2
    B = a.realizations
    # Batch of the engine: u + n_i, then 2u - d (cross), then n_i and d - u (res).
    nb = B + (1 if weights["cross"] else 0) + (B + 1 if weights["res"] else 0)

    one = NativeEngine(a.nside, j, batch=1, backend="gpu", device=a.device)
    many = NativeEngine(a.nside, j, batch=nb, backend="gpu", device=a.device)
    stats = Stats(many)
    loss_fn = DenoiseLoss(one.plan, a.nside, device=dev)
    if a.halfmission:
        # Pairs of the Delouis terms: targets S(d1,d2), S(d,u), S(T,d), then per
        # realisation S(u+ñ1,u+ñ2), S(u+ñ,u), S(T,u+ñ).
        xeng = NativeEngine(a.nside, j, batch=3 + 3 * B, backend="gpu", device=a.device, cross=True)
        xrun = {"engine": xeng, "generation": 0}
        xloss = DenoiseLoss(xeng.plan, a.nside, device=dev)
    nlev = int(math.log2(a.nside)) - 1

    def forward1(m):
        v, var = one.forward(m.detach().cpu().numpy()[None], moments=True)
        return torch.as_tensor(as_real(v[0]), device=dev), torch.as_tensor(var[0], device=dev)

    # Ergodic variances of the noise, for the `res` term (one training draw per field).
    noise_var = {f: forward1(n.draw(1)[0])[1] for f, n in noise.items()}

    def delouis(u, d, d1, d2, field, generator=None, test=False):
        """The three terms of Delouis et al. (2022) for `u` (with gradient) and
        the observation (d, d1, d2), on the full sky."""
        n, n1, n2 = noise[field].draw(B, test=test, generator=generator, hm=True)
        A = torch.cat([d1[None], d[None], T[None], u + n1, u + n, T.expand(B, -1)])
        Bm = torch.cat([d2[None], u[None], d[None], u + n2, u.expand(B, -1), u + n])
        # order of the realisations: [l1 x B, l2 x B, l3 x B]
        st = CrossStats.apply(A, Bm, xrun)
        var = xrun["var"]
        out = {}
        for t, name in enumerate(DELOUIS):
            target = st[t] if name == "l2" else st[t].detach()
            out[name] = xloss(st[3 + t * B:3 + (t + 1) * B], target, var[t])
        return out

    def losses(u, d, field, s_stats=None, generator=None, test=False, hm=None):
        """Terms of the loss for `u` `[npix]` (with gradient) and `d`; `hm`:
        the half-mission observations (d1, d2) for the Delouis terms."""
        if hm is not None:
            out = delouis(u, d, hm[0], hm[1], field, generator=generator, test=test)
            if weights["data"] == weights["res"] == weights["cross"] == 0:
                return out
        else:
            out = {}
        target, tvar = forward1(d)
        n = noise[field].draw(B, test=test, generator=generator)
        parts = [u + n]
        if weights["cross"]:
            parts += [(2 * u - d)[None]]
        if weights["res"]:
            parts += [n, (d - u)[None]]
        st = (s_stats or stats)(torch.cat(parts))
        out["data"], k = loss_fn(st[:B], target, tvar), B
        if weights["cross"]:
            out["cross"] = loss_fn(st[k:k + 1], target, tvar)
            k += 1
        if weights["res"]:
            out["res"] = loss_fn(st[k + B:k + B + 1], st[k:k + B].mean(0).detach(), noise_var[field])
        return out

    channels = [int(c) for c in a.channels.split(",")] if a.channels else None
    net = UNet(a.nside, levels, channels, conv=a.conv, gauge=a.gauge, ksz=a.ksz).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    nparams = sum(q.numel() for q in net.parameters())
    rms = {f: n.rms for f, n in noise.items()}
    print(f"nside {a.nside} J {j} coefficients {len(one.plan)} levels {levels} params {nparams} batch {nb} "
          f"loss {weights} noise {a.noise} rms {', '.join(f'{f} {v:.4g}' for f, v in rms.items())} uK "
          f"(train std {float(train_sky.std()):.4g}, test std {float(test_sky.std()):.4g})", flush=True)

    gen = torch.Generator(device=dev).manual_seed(12345)
    def observe(f, s, test=False, generator=None):
        """(d, (d1, d2) or None): an observation of the sky s of field f."""
        if not a.halfmission:
            return s + noise[f].draw(1, test=test, generator=generator)[0], None
        n, n1, n2 = noise[f].draw(1, test=test, generator=generator, hm=True)
        return s + n[0], (s + n1[0], s + n2[0])

    def real_obs(f):
        """(d, (d1, d2) or None): the PR3 full- and half-mission maps of field f."""
        hm = tuple(torch.as_tensor(np.load(os.path.join(a.data, f"planck353hm{h}_{f}_n{a.nside}.npy")), device=dev)
                   for h in (1, 2)) if a.halfmission else None
        return load(f), hm

    if a.real:
        fixed = {f: (s, *real_obs(f)) for f, s in ((a.train_field, train_sky), (a.test_field, test_sky))}
    else:
        fixed = {f: (s, *observe(f, s, test=True, generator=gen))
                 for f, s in ((a.train_field, train_sky), (a.test_field, test_sky))}
    fields = list(fixed)

    def evaluate(save=False):
        net.eval()
        out = {}
        with torch.no_grad():
            for f, (s, d, hm) in fixed.items():
                u = net(d[None], d.std())[0]
                ts, tv = forward1(s)
                us, _ = forward1(u)
                ds, _ = forward1(d)
                terms = {k: v.item() for k, v in losses(u, d, f, s_stats=lambda x: torch.as_tensor(
                    as_real(many.forward(x.cpu().numpy())[0]), device=dev), generator=gen, test=True, hm=hm).items()}
                out[f] = {
                    "pixel_rms_error_over_noise": float((u - s).square().mean().sqrt()) / rms[f],
                    "data_loss": terms.get("data", float("nan")),
                    "loss_terms": terms,
                    "stats_u_vs_sky": loss_fn.report(us, ts, tv),
                    "stats_d_vs_sky": loss_fn.report(ds, ts, tv),
                    "level_power_u_over_sky": [x / y for x, y in zip(level_powers(u.cpu(), nlev), level_powers(s.cpu(), nlev))],
                    "level_power_d_over_sky": [x / y for x, y in zip(level_powers(d.cpu(), nlev), level_powers(s.cpu(), nlev))],
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

    hist = {"step": [], "loss": [], **{t: [] for t in TERMS}}
    evals = {0: evaluate()}
    show(0, evals[0])
    started = time.time()
    for step in range(1, a.steps + 1):
        if a.real:
            f = fields[step % len(fields)]
            _, d, hm = fixed[f]
        else:
            f = a.train_field
            d, hm = observe(f, train_sky)
        u = net(d[None], d.std())[0]
        terms = losses(u, d, f, hm=hm)
        loss = sum(weights[k] * v for k, v in terms.items())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == 1:
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            for t in TERMS:
                hist[t].append(terms[t].item() if t in terms else None)
            print(f"step {step} loss {loss.item():.6g} " + " ".join(f"{k} {v.item():.4g}" for k, v in terms.items())
                  + f" {(time.time() - started) / step:.3f} s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evals[step] = evaluate(save=step == a.steps)
            show(step, evals[step])
            torch.save({"state": net.state_dict(), **net.config(), "j": j, "noise_rms": rms},
                       os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump({"args": vars(a), "j": j, "coefficients": len(one.plan), "parameters": nparams, "noise_rms": rms,
                   "loss_weights": weights, "seconds": time.time() - started, "loss_history": hist,
                   "evals": evals}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
