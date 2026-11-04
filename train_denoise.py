"""Train the HEALPix U-Net d -> u with the scattering denoising loss.

    python train_denoise.py --data DIR --out RUN [--nside 64 --train-field Q --test-field U
                            --noise-rel 0.5 --realizations 4 --steps 2000 --lr 1e-3 --device 0]

Sky maps: DIR/planck353_{field}_n{nside}.npy (NESTED, uK; `scripts/prepare_planck.py`).
Noise: white, `sigma = noise_rel * std(train sky)`, the same for every field;
with `--noise-std`, white with the per-pixel level of the PR3 simulations,
`sigma(p) = noise_scale * DIR/planck353_noise_std_{field}_n{nside}.npy`
(`scripts/noise_std.py`), each field with its own map.

Every step draws an observation `d = s sky + sigma n_0` of the training sky
(`s = +-1` with `--flip`), and trains on

    loss = sum_c w_c |mean_i S_c(f(d) + sigma n_i) - S_c(d)|^2,   i = 1..realizations

(`stlia.stl_torch.DenoiseLoss`): no pixel of the clean sky is used.  The
evaluation (fixed noise, train and test fields) compares `u = f(d)` with the
clean sky: pixel error relative to the noise, statistics of `u` and of `d`
against those of the sky (rms per family in units of the ergodic sigma of
the sky), and the variance of the 4-children details per level.

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--j", type=int, default=None)
    p.add_argument("--train-field", default="Q")
    p.add_argument("--test-field", default="U")
    p.add_argument("--noise-rel", type=float, default=0.5)
    p.add_argument("--noise-std", action="store_true", help="per-pixel sigma from the PR3 noise simulations")
    p.add_argument("--noise-scale", type=float, default=1.0, help="factor on the PR3 sigma map")
    p.add_argument("--realizations", type=int, default=4)
    p.add_argument("--flip", action="store_true", help="random sign of the training sky")
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--channels", default=None, help="comma-separated, one per level")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=250)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    j = a.j or int(math.log2(a.nside)) - 1
    dev = torch.device(f"cuda:{a.device}")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    def load(name):
        return torch.as_tensor(np.load(os.path.join(a.data, f"planck353_{name}_n{a.nside}.npy")), device=dev)

    train_sky, test_sky = load(a.train_field), load(a.test_field)
    if a.noise_std:
        sigmas = {f: a.noise_scale * load(f"noise_std_{f}") for f in (a.train_field, a.test_field)}
    else:
        sigmas = {f: torch.full_like(train_sky, a.noise_rel * float(train_sky.std()))
                  for f in (a.train_field, a.test_field)}
    rms = {f: float(s.square().mean().sqrt()) for f, s in sigmas.items()}
    npix = 12 * a.nside**2

    one = NativeEngine(a.nside, j, batch=1, backend="gpu", device=a.device)
    many = NativeEngine(a.nside, j, batch=a.realizations, backend="gpu", device=a.device)
    stats = Stats(many)
    loss_fn = DenoiseLoss(one.plan, a.nside, device=dev)
    nlev = int(math.log2(a.nside)) - 1

    def forward1(m):
        v, var = one.forward(m.detach().cpu().numpy()[None], moments=True)
        return torch.as_tensor(as_real(v[0]), device=dev), torch.as_tensor(var[0], device=dev)

    channels = [int(c) for c in a.channels.split(",")] if a.channels else None
    net = UNet(a.nside, a.levels, channels).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    nparams = sum(q.numel() for q in net.parameters())
    print(f"nside {a.nside} J {j} coefficients {len(one.plan)} params {nparams} "
          f"noise rms {', '.join(f'{f} {v:.4g}' for f, v in rms.items())} uK "
          f"(train std {float(train_sky.std()):.4g}, test std {float(test_sky.std()):.4g})", flush=True)

    gen = torch.Generator(device=dev).manual_seed(12345)
    fixed = {f: (s, s + sigmas[f] * torch.randn(npix, generator=gen, device=dev, dtype=s.dtype))
             for f, s in ((a.train_field, train_sky), (a.test_field, test_sky))}

    def evaluate(save=False):
        net.eval()
        out = {}
        with torch.no_grad():
            for f, (s, d) in fixed.items():
                u = net(d[None], d.std())[0]
                ts, tv = forward1(s)
                us, _ = forward1(u)
                ds, dv = forward1(d)
                x = u[None] + sigmas[f] * torch.randn(a.realizations, npix, generator=gen, device=dev, dtype=u.dtype)
                xs, _ = many.forward(x.cpu().numpy())
                data_loss = loss_fn(torch.as_tensor(as_real(xs), device=dev), ds, dv).item()
                out[f] = {
                    "pixel_rms_error_over_noise": float((u - s).square().mean().sqrt()) / rms[f],
                    "data_loss": data_loss,
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
            print(f"eval {step} {f}: pixel err/noise {e['pixel_rms_error_over_noise']:.3f} data {e['data_loss']:.4g} "
                  f"| u/d vs sky [sigma]: {fam}", flush=True)

    hist = {"step": [], "loss": []}
    evals = {0: evaluate()}
    show(0, evals[0])
    started = time.time()
    for step in range(1, a.steps + 1):
        s = train_sky * (1 if not a.flip or torch.rand(()) < 0.5 else -1)
        sigma = sigmas[a.train_field]
        d = s + sigma * torch.randn(npix, device=dev, dtype=s.dtype)
        target, tvar = forward1(d)
        u = net(d[None], d.std())
        x = u + sigma * torch.randn(a.realizations, npix, device=dev, dtype=u.dtype)
        loss = loss_fn(stats(x), target, tvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % a.log_every == 0 or step == 1:
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            print(f"step {step} loss {loss.item():.6g} {(time.time() - started) / step:.3f} s/step", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evals[step] = evaluate(save=step == a.steps)
            show(step, evals[step])
            torch.save({"state": net.state_dict(), **net.config(), "j": j, "noise_rms": rms},
                       os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump({"args": vars(a), "j": j, "coefficients": len(one.plan), "parameters": nparams, "noise_rms": rms,
                   "seconds": time.time() - started, "loss_history": hist, "evals": evals}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
