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

Writes RUN/model.pt, RUN/train.json, RUN/eval_{field}.npz (sky, d, u).
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

TERMS = ("data", "res", "cross")


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
            self.train = torch.stack([a.noise_scale * load(f"noise_{field}", k) for k in id_range(a.sims_train)])
            self.test = torch.stack([a.noise_scale * load(f"noise_{field}", k) for k in id_range(a.sims_test)])
        self.rms = float(self.draw(64, test=True).square().mean().sqrt())

    def draw(self, k, test=False, generator=None):
        if self.mode != "pr3sims":
            return self.sigma * torch.randn(k, self.npix, generator=generator, device=self.dev, dtype=self.sigma.dtype)
        pool = self.test if test else self.train
        idx = torch.randperm(len(pool), generator=generator, device=self.dev)[: min(k, len(pool))]
        idx = idx[torch.arange(k, device=self.dev) % len(idx)]  # reuse when k > pool (evaluation of small pools)
        sign = torch.randint(0, 2, (k, 1), generator=generator, device=self.dev, dtype=pool.dtype) * 2 - 1
        return pool[idx] * sign


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
    p.add_argument("--loss", default="data=1,res=1,cross=1", help="weights of the terms data, res, cross")
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
    nlev = int(math.log2(a.nside)) - 1

    def forward1(m):
        v, var = one.forward(m.detach().cpu().numpy()[None], moments=True)
        return torch.as_tensor(as_real(v[0]), device=dev), torch.as_tensor(var[0], device=dev)

    # Ergodic variances of the noise, for the `res` term (one training draw per field).
    noise_var = {f: forward1(n.draw(1)[0])[1] for f, n in noise.items()}

    def losses(u, d, field, s_stats=None, generator=None, test=False):
        """Terms of the loss for `u` `[npix]` (with gradient) and `d`."""
        target, tvar = forward1(d)
        n = noise[field].draw(B, test=test, generator=generator)
        parts = [u + n]
        if weights["cross"]:
            parts += [(2 * u - d)[None]]
        if weights["res"]:
            parts += [n, (d - u)[None]]
        st = (s_stats or stats)(torch.cat(parts))
        out, k = {"data": loss_fn(st[:B], target, tvar)}, B
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
    fixed = {f: (s, s + noise[f].draw(1, test=True, generator=gen)[0])
             for f, s in ((a.train_field, train_sky), (a.test_field, test_sky))}

    def evaluate(save=False):
        net.eval()
        out = {}
        with torch.no_grad():
            for f, (s, d) in fixed.items():
                u = net(d[None], d.std())[0]
                ts, tv = forward1(s)
                us, _ = forward1(u)
                ds, _ = forward1(d)
                terms = {k: v.item() for k, v in losses(u, d, f, s_stats=lambda x: torch.as_tensor(
                    as_real(many.forward(x.cpu().numpy())[0]), device=dev), generator=gen, test=True).items()}
                out[f] = {
                    "pixel_rms_error_over_noise": float((u - s).square().mean().sqrt()) / rms[f],
                    "data_loss": terms["data"],
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
        d = train_sky + noise[a.train_field].draw(1)[0]
        u = net(d[None], d.std())[0]
        terms = losses(u, d, a.train_field)
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
