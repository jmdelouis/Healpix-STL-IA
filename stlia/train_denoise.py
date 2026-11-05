"""Train the HEALPix U-Net d -> u with scattering denoising losses (full sky).

    python train_denoise.py --data DIR --out RUN [--nside 64 --train-field Q --test-field U
                            --noise pr3sims --realizations 4 --loss data=1,res=1,cross=1
                            --steps 3000 --lr 1e-3 --device 0]

Sky maps: DIR/planck353_{field}_n{nside}.npy (NESTED, uK; `scripts/prepare_planck.py`).

Noise (`--noise`), `noise_scale` times::

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
on the scattering covariances (`stlia.stl_torch.DenoiseLoss`, ergodic weights)::

  data   mean_i S(u + n_i)  vs  S(d)            (u plus noise looks like d)
  res    S(d - u)           vs  mean_i S(n_i)   (what is removed looks like noise)
  cross  S(2u - d)          vs  S(d)            (u - r looks like u + r)

and the full-sky terms of Delouis et al. (2022), with the cross statistics
S(a, b) of the GPU engine and half-mission observations d1 = sky + n1,
d2 = sky + n2 of the same FFP10 simulation as d (T: the 857 GHz map)::

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

from stlia.spectra import BinnedSpectra, NoiseSpectrumLoss, SpectrumLoss
from stlia.stl_torch import DenoiseLoss, Stats, as_real
from stlia.unet import UNet


TERMS = ("data", "res", "cross", "l1", "l2", "l3", "cl", "tcl", "rcl")
DELOUIS = ("l1", "l2", "l3")


def unbiased(plan, term):
    """`[ncoeff]` bool: the coefficients of `term` whose noise average equals the
    noise-free statistic, i.e. without noise bias.  The noise enters them only
    linearly, and is never under a modulus or a root:

    - data, Phi(u + n): the mean;
    - l1, Phi(u + n1, u + n2): S2, <a_z conj(b_z)>, bilinear, the two noises independent;
    - l2, Phi(u + n, u): S2, and S3 with the modulus of u (c1 = 1: modulus of b);
    - l3, Phi(T, u + n): S2, and S3 with the modulus of T (c1 = 0: modulus of a).

    S1 (root normalisation), S4 (moduli of both maps), the variance and the S3
    with the modulus of a noisy map are biased."""
    fam = np.array([c["family"] for c in plan])
    c1 = np.array([c.get("c1", -1) for c in plan])
    if term == "data":
        return fam == "mean"
    out = fam == "S2"
    if term == "l2":
        out |= (fam == "S3") & (c1 == 1)
    if term == "l3":
        out |= (fam == "S3") & (c1 == 0)
    return out


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
        n = values.shape[0]  # with K masks: [N, K, ncoeff], flattened to [N, K ncoeff]
        run["var"] = torch.as_tensor(var, device=a.device).reshape(n, -1)
        ctx.run, ctx.g, ctx.device, ctx.shape = run, g, a.device, values.shape
        return torch.from_numpy(as_real(values).reshape(n, -1, 2).copy()).to(a.device)

    @staticmethod
    def backward(ctx, grad):
        cot = grad.detach().cpu().numpy().reshape(ctx.shape + (2,))
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

    @classmethod
    def eb_pair(cls, nq, nu, eb):
        """The noises of the E and B maps (`stlia.polar.EBTransform` `eb`) of the
        (Q, U) noise simulations of `nq`, `nu` (pr3sims: the same simulations,
        in the same order, for Q and U)."""
        assert nq.mode == nu.mode == "pr3sims", "--pol-stats eb needs --noise pr3sims"
        out = [cls.__new__(cls) for _ in range(2)]
        for n in out:
            n.mode, n.dev, n.npix = nq.mode, nq.dev, nq.npix
        for split in ("train", "test"):
            pools = [eb.many(q, u) for q, u in zip(getattr(nq, split), getattr(nu, split))]
            for i, n in enumerate(out):
                setattr(n, split, [p[i] for p in pools])
        for n in out:
            n.rms = float(n.draw(64, test=True).square().mean().sqrt())
        return out

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


class ChiTerms:
    """The terms of `--loss` on the local chi_E + i chi_B = conj-eth^2 (Q + iU) of
    the spin-2 engine of STL-Dev-MPI (``NativeEngine(spin=2)``, branch
    claude/spin-eth: conj-eth^2 at the finest level, then the scalar pyramid of
    the engine) for a joint (Q, U) output `u`::

        data   mean_i S(chi(u + n_i))               vs  S(chi(d))
        l1     mean_i S(chi(u + ñ1_i), chi(u + ñ2_i)) vs  S(chi(d1), chi(d2))
        l2     mean_i S(chi(u + ñ_i), chi(u))         vs  S(chi(d), chi(u))
        l3     mean_i S(T, chi(u + ñ_i))              vs  S(T, chi(d))

    each statistic for E and for B (no E x B terms), flattened component-major,
    `[2 K ncoeff]`; T, the 857 GHz map, is scalar.  The Q and U noises of the
    same simulation are drawn together.  With frozen biases the model is the
    statistic of u alone plus a bias estimated at a fixed u, zero where
    ``unbiased(plan, term)``, as for the scalar terms."""

    def __init__(self, a, weights, fields, fixed, noise, nb, B, mk, T, dev):
        from stl_port.native import NativeEngine
        j = a.j or int(math.log2(a.nside)) - 1
        K = len(mk.get("masks", [None]))
        self.w, self.B, self.nb, self.M, self.bias_all, self.dev, self.T = weights, B, nb, a.bias_realizations, \
            a.bias_all, dev, T
        self.one = NativeEngine(a.nside, j, batch=1, backend="gpu", device=a.device, spin=2, **mk)
        self.many = NativeEngine(a.nside, j, batch=nb, backend="gpu", device=a.device, spin=2, **mk)
        self.stats = Stats(self.many)
        self.loss = self._loss(self.one, a, mk, K)
        self.exact = {"data": self._exact(self.one, "data", K)}
        self.pools = [noise[f].train for f in fields]  # [noise, hm1, hm2] per field, the same simulations
        qu = lambda i: torch.stack([fixed[f][i] for f in fields])
        self.d = qu(1)
        self.target, self.tvar = self.forward1(self.d)
        self.cross = any(weights[t] for t in DELOUIS)
        if self.cross:
            self.d1, self.d2 = (torch.stack([fixed[f][2][h] for f in fields]) for h in (0, 1))
            # pairs: spin x spin, [(d1, d2), (d, u), l1 x B, l2 x B]; scalar T x spin, [(T, d), l3 x B]
            self.nqq, self.ntq = 2 + 2 * B, 1 + B
            self.xqq = {"engine": NativeEngine(a.nside, j, batch=self.nqq, backend="gpu", device=a.device, spin=2,
                                               cross=True, spin_a=2, **mk), "generation": 0}
            self.xtq = {"engine": NativeEngine(a.nside, j, batch=self.ntq, backend="gpu", device=a.device, spin=2,
                                               cross=True, spin_a=0, **mk), "generation": 0}
            plan = self.xqq["engine"].plan
            self.xloss = self._loss(self.xqq["engine"], a, mk, K)
            self.exact.update({t: self._exact(self.xqq["engine"], t, K) for t in DELOUIS})
        print(f"statistics of chi_E, chi_B (spin-2 engine): {2 * K * len(self.one.plan)} coefficients"
              + (f", cross {2 * K * len(plan)}" if self.cross else ""), flush=True)

    def _loss(self, eng, a, mk, K):
        npix = np.tile(np.asarray(eng.native_pixels(), dtype=np.float64).ravel(), 2) if mk else None
        return DenoiseLoss(eng.plan, a.nside, device=self.dev, masks=2 * K, npix=npix)

    def _exact(self, eng, term, K):
        return torch.as_tensor(np.tile(unbiased(eng.plan, term), 2 * K), device=self.dev)

    def forward1(self, qu):
        """Statistics and moments of one (Q, U) map `[2, npix]`, without gradient."""
        v, var = self.one.forward(qu.detach().cpu().numpy()[None], moments=True)
        return (torch.as_tensor(as_real(v[0]).reshape(-1, 2), device=self.dev),
                torch.as_tensor(var[0], device=self.dev).reshape(-1))

    def draw(self, k, generator=None):
        """(n, n1, n2), each `[k, 2, npix]`: (Q, U) noises of k training simulations
        (full and half missions of the same simulations), random signs."""
        idx = torch.randint(len(self.pools[0][0]), (k,), generator=generator, device=self.dev)
        sign = torch.randint(0, 2, (k, 1, 1), generator=generator, device=self.dev).to(self.d.dtype) * 2 - 1
        return tuple(torch.stack([p[h][idx] for p in self.pools], 1) * sign for h in range(len(self.pools[0])))

    def auto(self, x):
        """Statistics with gradient of `x` `[k <= nb, 2, npix]` -> `[k, 2 K ncoeff, 2]`."""
        return self.stats(_pad(x, self.nb))[:len(x)]

    def pairs(self, run, n, a_, b_):
        """Cross statistics with gradient of `k <= n` pairs -> (`[k, 2 K ncoeff, 2]`, moments)."""
        k = len(a_)
        st = CrossStats.apply(_pad(a_, n), _pad(b_, n), run)
        return st[:k], run["var"][:k]

    def terms(self, u, bias=None):
        """Terms of `--loss` for the joint output `u` `[2, npix]`: live draws, or the frozen `bias`."""
        out = {}
        T, B = self.T, self.B
        if self.w["data"]:
            st = self.auto(u[None] + self.draw(B)[0]) if bias is None else self.auto(u[None]) + bias["data"]
            out["data"] = self.loss(st, self.target, self.tvar)
        if not self.cross:
            return out
        if bias is None:
            n, n1, n2 = self.draw(B)
            st, var = self.pairs(self.xqq, self.nqq, torch.cat([self.d1[None], self.d[None], u + n1, u + n]),
                                 torch.cat([self.d2[None], u[None], u + n2, u.expand(B, -1, -1)]))
            sq, vq = self.pairs(self.xtq, self.ntq, T.expand(1 + B, -1), torch.cat([self.d[None], u + n]))
            out["l1"] = self.xloss(st[2:2 + B], st[0].detach(), var[0])
            out["l2"] = self.xloss(st[2 + B:2 + 2 * B], st[1], var[1])
            out["l3"] = self.xloss(sq[1:], sq[0].detach(), vq[0])
        else:
            st, var = self.pairs(self.xqq, self.nqq, torch.stack([self.d1, self.d, u]), torch.stack([self.d2, u, u]))
            sq, vq = self.pairs(self.xtq, self.ntq, T.expand(2, -1), torch.stack([self.d, u]))
            out["l1"] = self.xloss((st[2] + bias["l1"])[None], st[0].detach(), var[0])
            out["l2"] = self.xloss((st[2] + bias["l2"])[None], st[1], var[1])
            out["l3"] = self.xloss((sq[1] + bias["l3"])[None], sq[0].detach(), vq[0])
        return {t: v for t, v in out.items() if self.w[t]}

    def estimate_bias(self, u, generator=None):
        """Noise biases at the fixed (Q, U) map `u`, over `--bias-realizations` draws."""
        M, T = self.M, self.T
        bias = {}
        with torch.no_grad():
            acc = 0
            for k0 in range(0, M, self.nb):
                k = min(self.nb, M - k0)
                acc = acc + self.auto(u[None] + self.draw(k, generator)[0]).sum(0)
            bias["data"] = acc / M - self.auto(u[None])[0]
            if self.cross:
                uu = self.pairs(self.xqq, self.nqq, u[None], u[None])[0][0]
                tu = self.pairs(self.xtq, self.ntq, T[None], u[None])[0][0]
                acc = {t: 0 for t in DELOUIS}
                per = self.nqq // 2
                for k0 in range(0, M, per):
                    k = min(per, M - k0)
                    n, n1, n2 = self.draw(k, generator)
                    st = self.pairs(self.xqq, self.nqq, torch.cat([u + n1, u + n]),
                                    torch.cat([u + n2, u.expand(k, -1, -1)]))[0]
                    acc["l1"] = acc["l1"] + st[:k].sum(0)
                    acc["l2"] = acc["l2"] + st[k:].sum(0)
                for k0 in range(0, M, self.ntq):
                    k = min(self.ntq, M - k0)
                    n = self.draw(k, generator)[0]
                    acc["l3"] = acc["l3"] + self.pairs(self.xtq, self.ntq, T.expand(k, -1), u + n)[0].sum(0)
                bias["l1"] = acc["l1"] / M - uu
                bias["l2"] = acc["l2"] / M - uu
                bias["l3"] = acc["l3"] / M - tu
            if not self.bias_all:
                for t in bias:
                    bias[t][self.exact[t]] = 0
        return bias


def _pad(x, n):
    """`x` `[k <= n, ...]` -> `[n, ...]`, completed with copies of x[0] (zero cotangents)."""
    return torch.cat([x, x[:1].expand((n - len(x),) + x.shape[1:])]) if len(x) < n else x


def parser():
    """Options of ``stlia-train-denoise`` (``python train_denoise.py``)."""
    p = argparse.ArgumentParser(prog="stlia-train-denoise",
                                description="Train the full-sky U-Net d -> u with scattering losses (GPU).")
    p.add_argument("--data", required=True, help="directory of the degraded maps (planck353_*_n<nside>.npy)")
    p.add_argument("--out", required=True, help="run directory: log, model.pt, train.json, eval_<field>.npz")
    p.add_argument("--nside", type=int, default=64, help="HEALPix resolution of the maps and of the network")
    p.add_argument("--j", type=int, default=None, help="scales of the statistics (default: log2(nside) - 1)")
    p.add_argument("--train-field", default="Q", help="field of the training sky (I, Q or U)")
    p.add_argument("--test-field", default="U", help="field of the test sky, never seen in training")
    p.add_argument("--noise", default="pr3sims", choices=["uniform", "pr3white", "pr3sims"],
                   help="noise model: uniform white, white with the PR3 per-pixel level, or FFP10 draws")
    p.add_argument("--noise-rel", type=float, default=0.5, help="uniform: sigma / std(train sky)")
    p.add_argument("--noise-scale", type=float, default=1.0, help="factor on every noise model")
    p.add_argument("--sims-train", default="10-49", help="FFP10 simulations of the training draws (e.g. 10-49 or 1,3,5)")
    p.add_argument("--sims-test", default="0-9", help="FFP10 simulations of the evaluation only")
    p.add_argument("--realizations", type=int, default=4, help="noise realisations n_i per step (B)")
    p.add_argument("--real", action="store_true",
                   help="denoise the PR3 maps themselves: d and (d1, d2) are the data, both fields are trained")
    p.add_argument("--loss", default="data=1,res=1,cross=1",
                   help="weights of the terms data, res, cross, of the Delouis et al. (2022) terms l1, l2, l3, "
                        "of the binned half-mission cross spectrum cl, of the cross spectrum tcl of the 857 GHz map "
                        "with u against that with d, and of the spectrum rcl of d - u against that of the noise "
                        "(tcl and rcl: spectra of the fields of the statistics, Q and U or, with --pol-stats eb, "
                        "E and B; the --cl-* options apply)")
    p.add_argument("--stat-masks", default=None,
                   help="scattering statistics averaged under galactic masks: latitude cuts b0 in degrees, "
                        "comma-separated, each mask keeping abs(b) > b0 (0: full sky); every coefficient then "
                        "enters once per mask, with its own ergodic weight (STL-Dev-MPI >= 9038114)")
    p.add_argument("--cl-masks", default="0,30,45",
                   help="cl: galactic latitude cuts b0 in degrees, comma-separated: the masks keep abs(b) > b0 (0: full sky)")
    p.add_argument("--cl-lmin", type=int, default=10, help="cl: first multipole")
    p.add_argument("--rcl-masks", default="30,45,60",
                   help="rcl: galactic latitude cuts of its masks; on real data its target C(d, d) - C(d1, d2) is "
                        "dominated in the Galactic plane by the difference between d and the mean of d1 and d2")
    p.add_argument("--rcl-lmin", type=int, default=40, help="rcl: first multipole")
    p.add_argument("--cl-lmax", type=int, default=None, help="cl: last multipole (default 2 nside - 1)")
    p.add_argument("--cl-width", type=int, default=30, help="cl: width of the multipole bins")
    p.add_argument("--cl-apod", type=float, default=5.0, help="cl: FWHM of the mask apodisation (degrees)")
    p.add_argument("--cl-spin", type=int, default=0, choices=[0, 2],
                   help="cl: 0, spectra of Q and U separately; 2, EE and BB spectra of (Q, U) (needs --joint and "
                        "the stl backend)")
    p.add_argument("--joint", action="store_true",
                   help="--real: one network denoises Q and U together (UNet(maps=2)); every step trains both")
    p.add_argument("--pol-stats", default="qu", choices=["qu", "eb", "chi"],
                   help="--joint: fields of the loss terms; qu, the Q and U maps; eb, the E and B maps of (Q, U) "
                        "(stlia.polar, exact spin-2 transforms of the full sky), which do not depend on the frame "
                        "of Q and U; with cl (spin 0), the spectra of the E and B maps; chi, the local "
                        "chi_E + i chi_B = conj-eth^2 (Q + iU) of the spin-2 engine of STL-Dev-MPI (NativeEngine "
                        "spin=2, branch claude/spin-eth f4cf030 or later), with masks; terms data, l1, l2, l3 (l3: scalar 857 GHz x "
                        "chi), and cl with --cl-spin 2")
    p.add_argument("--cl-backend", default="auto", choices=["auto", "torch", "stl"],
                   help="cl: spectra by the SpectrumEngine of STL-Dev-MPI (stl), by the transforms of stlia.sht "
                        "(torch), or by the engine when the installed STL-Dev-MPI has it (auto)")
    p.add_argument("--levels", type=int, default=None, help="levels of the U-Net (default: down to nside 8)")
    p.add_argument("--channels", default=None, help="channels per level, comma-separated (default: 32,64,96,128,...)")
    p.add_argument("--cond", default="none", choices=["none", "log857", "sigma", "log857+sigma"],
                   help="conditioning maps of the U-Net, standardised, given at the input and to every decoder "
                        "level: log857, the log of the 857 GHz intensity (local dust intensity); sigma, the log of "
                        "the per-pixel noise level, the rms over the training simulations of the noise of the "
                        "fields (sqrt of the mean of the Q and U variances: the local noise level)")
    p.add_argument("--conv", default="sphere", choices=["sphere", "faces"],
                   help="STL stencil on the sphere, or planar convolutions on the base faces (comparison only)")
    p.add_argument("--gauge", default="phi", choices=["phi", "cosmo"], help="orientation of the kernels")
    p.add_argument("--ksz", type=int, default=3, help="kernel size (ksz x ksz taps)")
    p.add_argument("--steps", type=int, default=3000, help="Adam steps (cosine learning-rate schedule)")
    p.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    p.add_argument("--device", type=int, default=0, help="CUDA device of the network and of the engine")
    p.add_argument("--seed", type=int, default=0, help="seed of torch and numpy")
    p.add_argument("--log-every", type=int, default=10, help="steps between two log lines")
    p.add_argument("--eval-every", type=int, default=250, help="steps between two evaluations (and checkpoints)")
    p.add_argument("--optimizer", default="adam", choices=["adam", "lbfgs"],
                   help="adam: train the U-Net; lbfgs: no network, minimise the same losses over the pixels of u "
                        "for the fixed observations (the per-map method of Delouis et al. 2022, the reference)")
    p.add_argument("--lbfgs-rounds", type=int, default=10,
                   help="lbfgs: rounds; the noise realisations are redrawn between rounds and fixed within one")
    p.add_argument("--lbfgs-iter", type=int, default=50, help="lbfgs: L-BFGS iterations per round")
    p.add_argument("--lbfgs-history", type=int, default=10, help="lbfgs: L-BFGS history size")
    p.add_argument("--bias", default="live", choices=["live", "frozen"],
                   help="live: <Phi(u + n_i)> over --realizations draws at every step, gradient through it; "
                        "frozen: Phi(u) + B, the noise bias B estimated at a fixed u and kept constant "
                        "(Delouis et al. 2022); needs fixed observations (--real or --optimizer lbfgs)")
    p.add_argument("--bias-realizations", type=int, default=40, help="frozen: noise draws of a bias estimate")
    p.add_argument("--bias-all", action="store_true",
                   help="frozen: estimate a bias for every coefficient, also those whose noise average is the "
                        "noise-free statistic (S2 of the cross terms, ...; see unbiased()); by default their "
                        "bias is zero")
    p.add_argument("--bias-every", type=int, default=50,
                   help="frozen, network: steps between two bias estimates (L-BFGS: one per round)")
    p.add_argument("--bias-warmup", type=int, default=0,
                   help="frozen: the first steps (L-BFGS: rounds) with live biases, so that the first frozen "
                        "estimate is made at a u already denoised")
    p.add_argument("--init", default=None,
                   help="model.pt of a previous run: start from its weights (same network options)")
    return p


def main(argv=None):
    """Run one training from the command line (options: :func:`parser`)."""
    from stl_port.native import NativeEngine  # STL-Dev-MPI: imported here, so that the package imports without it

    a = parser().parse_args(argv)
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
    a.halfmission = any(weights[t] for t in DELOUIS + ("cl",))
    spec_terms = weights["cl"] or weights["tcl"] or weights["rcl"]
    if spec_terms:
        if a.cl_backend == "auto":
            import stl_port.native
            a.cl_backend = "stl" if hasattr(stl_port.native, "SpectrumEngine") else "torch"
        assert a.cl_spin == 0 or (a.joint and a.cl_backend == "stl"), "--cl-spin 2 needs --joint and the stl backend"
        print(f"cl: spectra (spin {a.cl_spin}) by the {a.cl_backend} backend", flush=True)
        spectra = BinnedSpectra(a.nside, masks=[float(b) for b in a.cl_masks.split(",")], lmin=a.cl_lmin,
                                lmax=a.cl_lmax, width=a.cl_width, apod=a.cl_apod, device=f"cuda:{a.device}",
                                backend=a.cl_backend, spin=a.cl_spin)
        cl_cache = {}
        rspectra = BinnedSpectra(a.nside, masks=[float(b) for b in a.rcl_masks.split(",")], lmin=a.rcl_lmin,
                                 lmax=a.cl_lmax, width=a.cl_width, apod=a.cl_apod, device=f"cuda:{a.device}",
                                 backend=a.cl_backend) if weights["rcl"] else None

        def cl_loss(field, d1, d2):
            """Spectrum loss of the half-mission pair (d1, d2) (cached for fixed observations)."""
            key = (field, d1.data_ptr(), d2.data_ptr())
            if key not in cl_cache:
                if len(cl_cache) > 8:
                    cl_cache.clear()
                cl_cache[key] = SpectrumLoss(spectra, d1, d2)
            return cl_cache[key]
    scalar_cl = weights["cl"] and a.cl_spin == 0  # per field; spin 2: one term for (Q, U)
    assert not (weights["tcl"] or weights["rcl"]) or (a.cl_spin == 0 and a.pol_stats != "chi"),         "tcl and rcl: spectra of scalar maps (--cl-spin 0, --pol-stats qu or eb)"
    assert not a.joint or (a.real and a.optimizer == "adam"), "--joint: --real and the adam optimizer"
    assert a.pol_stats != "eb" or (a.joint and a.cl_spin == 0), \
        "--pol-stats eb: --joint, and cl on the E and B maps (--cl-spin 0)"
    assert a.pol_stats != "chi" or (a.joint and not any(weights[t] for t in ("res", "cross"))
                                    and (not weights["cl"] or a.cl_spin == 2)), \
        "--pol-stats chi: --joint, the terms data, l1, l2, l3, and cl with --cl-spin 2"
    assert a.bias_warmup == 0 or a.bias == "frozen", "--bias-warmup: --bias frozen"
    if a.halfmission or weights["tcl"]:
        T = torch.as_tensor(np.load(os.path.join(a.data, f"planck857_I_n{a.nside}.npy")), device=dev)
    a.noise_rel_std = a.noise_rel * float(train_sky.std())
    noise = {f: Noise(a, f, s, load, dev) for f, s in ((a.train_field, train_sky), (a.test_field, test_sky))}
    npix = 12 * a.nside**2
    B = a.realizations
    # Batch of the engine: u + n_i, then 2u - d (cross), then n_i and d - u (res).
    nb = B + (1 if weights["cross"] else 0) + (B + 1 if weights["res"] else 0)

    mk = {}
    if a.stat_masks:
        # masked averages of the full-sky statistics (STL-Dev-MPI >= 9038114): K sets of coefficients
        from stlia.spectra import galactic_mask
        cuts = [float(b) for b in a.stat_masks.split(",")]
        mk = {"masks": [np.flatnonzero(galactic_mask(a.nside, b, apod=0) > 0.5) for b in cuts]}
    one = NativeEngine(a.nside, j, batch=1, backend="gpu", device=a.device, **mk)
    many = NativeEngine(a.nside, j, batch=nb, backend="gpu", device=a.device, **mk)
    stats = Stats(many)
    K = len(mk.get("masks", [None]))
    loss_fn = DenoiseLoss(one.plan, a.nside, device=dev, masks=K, npix=one.native_pixels() if mk else None)
    if a.halfmission:
        # Pairs of the Delouis terms: targets S(d1,d2), S(d,u), S(T,d), then per
        # realisation S(u+ñ1,u+ñ2), S(u+ñ,u), S(T,u+ñ).
        xeng = NativeEngine(a.nside, j, batch=3 + 3 * B, backend="gpu", device=a.device, cross=True, **mk)
        xrun = {"engine": xeng, "generation": 0}
        xloss = DenoiseLoss(xeng.plan, a.nside, device=dev, masks=K, npix=xeng.native_pixels() if mk else None)
    # coefficients without noise bias (frozen biases: B = 0 there, unless --bias-all), K masks each
    exact = {t: torch.as_tensor(np.tile(unbiased(one.plan if t == "data" else xeng.plan, t), K), device=dev)
             for t in ("data",) + (DELOUIS if a.halfmission else ())}
    nlev = int(math.log2(a.nside)) - 1

    def forward1(m):
        v, var = one.forward(m.detach().cpu().numpy()[None], moments=True)
        return (torch.as_tensor(as_real(v[0]).reshape(-1, 2), device=dev),
                torch.as_tensor(var[0], device=dev).reshape(-1))

    # Ergodic variances of the noise, for the `res` term (one training draw per field).
    noise_var = {f: forward1(n.draw(1)[0])[1] for f, n in noise.items()}

    spec_cache, rcl_cache = {}, {}

    def spectrum_terms(u, d, field, hm=None):
        """The terms tcl and rcl of `u` (with gradient) and `d`: spectra of u
        alone, without noise draws or biases (live and frozen alike)."""
        out = {}
        if weights["tcl"]:
            # the noise is independent of T: <C(T, d)> = C(T, s)
            key = ("tcl", field, d.data_ptr())
            if key not in spec_cache:
                if len(spec_cache) > 8:
                    spec_cache.clear()
                spec_cache[key] = SpectrumLoss(spectra, T, d)
            out["tcl"] = spec_cache[key](spectra(T, u))
        if weights["rcl"]:
            if field not in rcl_cache:
                # real data: the noise of d itself, C(d, d) - C(d1, d2); otherwise the mean over
                # the noise model
                if a.real and hm is not None:
                    rcl_cache[field] = NoiseSpectrumLoss.from_halves(rspectra, d, hm[0], hm[1])
                else:
                    pool = noise[field]
                    sims = pool.train[0] if pool.mode == "pr3sims" else pool.draw(40)
                    rcl_cache[field] = NoiseSpectrumLoss(rspectra, sims)
            out["rcl"] = rcl_cache[field](rspectra(d - u))
        return out

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
        out = {}
        if hm is not None and any(weights[t] for t in DELOUIS):
            out.update(delouis(u, d, hm[0], hm[1], field, generator=generator, test=test))
        if hm is not None and scalar_cl:
            # the half-mission noises are independent: <C(u + n1, u + n2)> = C(u, u), no bias to model
            out["cl"] = cl_loss(field, hm[0], hm[1])(spectra(u, u))
        out.update(spectrum_terms(u, d, field, hm))
        if weights["data"] == weights["res"] == weights["cross"] == 0:
            return out
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

    # -- Frozen noise biases (--bias frozen), as in Delouis et al. (2022): the model of a term
    # is the statistic of u alone plus a bias B = <Phi(u0 + n_i)> - Phi(u0) estimated at a fixed
    # u0 with many realisations and kept constant (no gradient) until the next estimate.
    def padded(x, batch):
        """`x` `[k, npix]` -> `[batch, npix]`, completed with copies of x[0] (their
        cotangents are zero, so they add nothing to the gradient)."""
        return torch.cat([x, x[:1].expand(batch - len(x), -1)]) if len(x) < batch else x

    def auto_stats(x):
        """Statistics with gradient of `x` `[k <= nb, npix]` -> `[k, ncoeff, 2]`."""
        return stats(padded(x, nb))[:len(x)]

    def cross_stats(a_, b_):
        """Cross statistics with gradient of the pairs `[k <= 3 + 3B, npix]`."""
        n = len(a_)
        st = CrossStats.apply(padded(a_, 3 + 3 * B), padded(b_, 3 + 3 * B), xrun)
        return st[:n], xrun["var"][:n]

    res_target = {}

    def noise_stats(field, hm=None):
        """Target of the term res with frozen biases, computed once: on real data the
        statistics of the half-mission difference, the noise of the data themselves;
        otherwise the mean over `--bias-realizations` training noise draws of `field`."""
        if field not in res_target:
            with torch.no_grad():
                if a.real and hm is not None:
                    res_target[field] = auto_stats((0.5 * (hm[0] - hm[1]))[None])[0]
                else:
                    M, acc = a.bias_realizations, 0
                    for k0 in range(0, M, nb):
                        acc = acc + auto_stats(noise[field].draw(min(nb, M - k0))).sum(0)
                    res_target[field] = acc / M
        return res_target[field]

    def estimate_bias(u, field, generator=None):
        """Noise biases of the terms of `--loss` at the fixed map `u` (no gradient),
        averaged over `--bias-realizations` training draws."""
        M = a.bias_realizations
        bias = {}
        with torch.no_grad():
            if weights["data"]:
                acc = 0
                for k0 in range(0, M, nb):
                    k = min(nb, M - k0)
                    n = noise[field].draw(k, generator=generator)
                    acc = acc + auto_stats(u + n).sum(0)
                bias["data"] = acc / M - auto_stats(u[None])[0]
            if any(weights[t] for t in DELOUIS):
                st0, _ = cross_stats(torch.stack([u, T]), torch.stack([u, u]))
                uu, tu = st0[0], st0[1]  # Phi(u, u), Phi(T, u)
                acc = {t: 0 for t in DELOUIS}
                per = (3 + 3 * B) // 3
                for k0 in range(0, M, per):
                    k = min(per, M - k0)
                    n, n1, n2 = noise[field].draw(k, generator=generator, hm=True)
                    st, _ = cross_stats(torch.cat([u + n1, u + n, T.expand(k, -1)]),
                                        torch.cat([u + n2, u.expand(k, -1), u + n]))
                    for t, name in enumerate(DELOUIS):
                        acc[name] = acc[name] + st[t * k:(t + 1) * k].sum(0)
                bias["l1"] = acc["l1"] / M - uu
                bias["l2"] = acc["l2"] / M - uu
                bias["l3"] = acc["l3"] / M - tu
            if not a.bias_all:
                for t in bias:
                    bias[t][exact[t]] = 0  # no noise bias there: the estimate would only add scatter
        return bias

    def frozen_losses(u, d, field, bias, hm=None):
        """Terms of `--loss` with frozen biases: Phi(u) + B against the targets of
        `losses` (same weights), without noise draws."""
        out = {}
        if weights["data"] or weights["res"]:
            # one forward for u and d - u: the engine keeps the state of its last forward only
            ntarget = noise_stats(field, hm) if weights["res"] else None
            st = auto_stats(torch.stack([u, d - u]))
        if weights["data"]:
            target, tvar = forward1(d)
            out["data"] = loss_fn((st[0] + bias["data"])[None], target, tvar)
        if weights["res"]:
            # Phi(d - u) against the mean statistics of the noise: no noise draw on the model side
            out["res"] = loss_fn(st[1:2], ntarget, noise_var[field])
        if scalar_cl:
            out["cl"] = cl_loss(field, hm[0], hm[1])(spectra(u, u))  # cross spectrum: unbiased, no bias
        out.update(spectrum_terms(u, d, field, hm))
        if any(weights[t] for t in DELOUIS):
            d1, d2 = hm
            st, var = cross_stats(torch.stack([d1, d, T, u, T]), torch.stack([d2, u, d, u, u]))
            model = {"l1": st[3], "l2": st[3], "l3": st[4]}
            for t, name in enumerate(DELOUIS):
                target = st[t] if name == "l2" else st[t].detach()
                out[name] = xloss((model[name] + bias[name])[None], target, var[t])
        return out

    channels = [int(c) for c in a.channels.split(",")] if a.channels else None
    conds = []
    if "log857" in a.cond:
        i857 = np.load(os.path.join(a.data, f"planck857_I_n{a.nside}.npy"))
        conds.append(np.log(np.clip(i857, np.percentile(i857[i857 > 0], 0.1), None)))
    if "sigma" in a.cond:
        # local noise level: rms over the training simulations, Q and U variances averaged
        assert a.noise == "pr3sims", "--cond sigma: --noise pr3sims"
        var = torch.stack([n.train[0].square().mean(0) for n in noise.values()]).mean(0)
        conds.append(0.5 * np.log(var.cpu().numpy()))
    cond = None
    if conds:
        c = np.stack([(x - x.mean()) / x.std() for x in conds])
        cond = torch.as_tensor(c, device=dev, dtype=torch.float32)[None]
    net = UNet(a.nside, levels, channels, conv=a.conv, gauge=a.gauge, ksz=a.ksz,
               cond=len(conds), maps=2 if a.joint else 1).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)["state"])
        print(f"weights from {a.init}", flush=True)
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

    eb, qu_fields, fixed_qu, last_qu = None, [], {}, {}
    if a.joint:
        # one network for (Q, U): d2 [2, npix], and the EE/BB spectrum loss of (Q, U) (--cl-spin 2)
        d2 = torch.stack([fixed[f][1] for f in fields])
        pol_loss = None
        if weights["cl"] and a.cl_spin == 2:
            hm1, hm2 = (torch.stack([fixed[f][2][h] for f in fields]) for h in (0, 1))
            pol_loss = SpectrumLoss(spectra, hm1, hm2)
        if a.pol_stats == "eb":
            # the terms on (E, B): the observations, the noise pools and their ergodic
            # variances are transformed once; the network still maps (Q, U) to (Q, U)
            from stlia.polar import EBTransform
            eb = EBTransform(a.nside, device=dev)
            qu_fields, fixed_qu, noise_qu = list(fields), dict(fixed), dict(noise)
            fields[:] = ["E", "B"]
            s_eb, d_eb = (eb(torch.stack([fixed_qu[f][i] for f in qu_fields])) for i in (0, 1))
            hm_eb = [eb(torch.stack([fixed_qu[f][2][h] for f in qu_fields])) for h in (0, 1)] \
                if a.halfmission else None
            fixed.clear()
            for i, f in enumerate(fields):
                fixed[f] = (s_eb[i], d_eb[i], (hm_eb[0][i], hm_eb[1][i]) if hm_eb else None)
            noise.clear()
            noise.update(zip(fields, Noise.eb_pair(noise_qu[qu_fields[0]], noise_qu[qu_fields[1]], eb)))
            noise_var.clear()
            noise_var.update({f: forward1(n.draw(1)[0])[1] for f, n in noise.items()})
            rms.clear()
            rms.update({f: n.rms for f, n in noise.items()})
            print(f"statistics of the E and B maps; noise rms E {rms['E']:.4g}, B {rms['B']:.4g} uK", flush=True)
        chi = None
        if a.pol_stats == "chi":
            chi = ChiTerms(a, weights, fields, fixed, noise, nb, B, mk, T if a.halfmission else None, dev)

        def joint_outputs():
            with torch.no_grad():
                u2 = net(d2[None], d2.std(), cond=cond)[0]
            if eb is not None:
                last_qu.update({f: u2[i] for i, f in enumerate(qu_fields)})
                u2 = eb(u2)
            return {f: u2[i] for i, f in enumerate(fields)}

        def joint_terms(u2, bias=None):
            """Terms of `--loss` for the joint output `u2` `[2, npix]`, in parts: the
            scattering terms of each field, then the EE/BB term of the pair.  Each
            part must be differentiated before the next one is computed: the
            engines keep the state of their last forward only."""
            if chi is not None:
                yield chi.terms(u2, bias)
            else:
                for i, f in enumerate(fields):
                    _, d, hm = fixed[f]
                    yield frozen_losses(u2[i], d, f, bias[f], hm=hm) if bias else losses(u2[i], d, f, hm=hm)
            if pol_loss is not None:
                yield {"cl": pol_loss(spectra(u2, u2))}  # cross spectra of the half missions: no bias

    def evaluate(save=False, outputs=None):
        """Evaluate the network, or the maps `outputs[field]` (L-BFGS)."""
        net.eval()
        out = {}
        with torch.no_grad():
            for f, (s, d, hm) in fixed.items():
                u = net(d[None], d.std(), cond=cond)[0] if outputs is None else outputs[f]
                ts, tv = forward1(s)
                us, _ = forward1(u)
                ds, _ = forward1(d)
                terms = {k: v.item() for k, v in losses(u, d, f, s_stats=lambda x: torch.as_tensor(
                    as_real(many.forward(x.cpu().numpy())[0]).reshape(len(x), -1, 2), device=dev), generator=gen, test=True, hm=hm).items()}
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
            if save:  # --pol-stats eb: the (Q, U) maps as well, for the figures
                for f, u in last_qu.items():
                    s, d, _ = fixed_qu[f]
                    np.savez(os.path.join(a.out, f"eval_{f}.npz"), sky=s.cpu().numpy(), d=d.cpu().numpy(), u=u.cpu().numpy())
        net.train()
        return out

    def show(step, ev):
        for f, e in ev.items():
            fam = " ".join(f"{k} {v:.3g}/{e['stats_d_vs_sky'][k]:.3g}" for k, v in e["stats_u_vs_sky"].items())
            terms = " ".join(f"{k} {v:.4g}" for k, v in e["loss_terms"].items())
            print(f"eval {step} {f}: pixel err/noise {e['pixel_rms_error_over_noise']:.3f} {terms} "
                  f"| u/d vs sky [sigma]: {fam}", flush=True)

    frozen = None
    if a.bias == "frozen":
        assert a.real or a.optimizer == "lbfgs", "--bias frozen needs fixed observations (--real or --optimizer lbfgs)"
        frozen = (estimate_bias, frozen_losses)
    if a.joint:
        if not frozen:
            bias_all = None
        elif chi is not None:
            bias_all = chi.estimate_bias
        else:
            def bias_all(z, generator):
                return {f: estimate_bias(z[i].detach(), f, generator=generator) for i, f in enumerate(fields)}
        return run_joint(a, net, d2, cond, fields, joint_outputs, joint_terms, bias_all,
                         weights, evaluate, show, {"j": j, "coefficients": len(one.plan), "noise_rms": rms}, eb)
    if a.optimizer == "lbfgs":
        return run_lbfgs(a, fixed, fields, losses, weights, evaluate, show, dev,
                         {"j": j, "coefficients": len(one.plan), "noise_rms": rms}, frozen)
    bias = {}
    bias_gen = torch.Generator(device=dev).manual_seed(a.seed + 777)

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
        u = net(d[None], d.std(), cond=cond)[0]
        if frozen and step > a.bias_warmup:
            if f not in bias or (step - 1 - a.bias_warmup) % a.bias_every < len(fields):
                bias[f] = estimate_bias(u.detach(), f, generator=bias_gen)  # at the current network output
            terms = frozen_losses(u, d, f, bias[f], hm=hm)
        else:
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
            # write then rename: a failed write leaves the previous checkpoint intact
            path = os.path.join(a.out, "model.pt")
            torch.save({"state": net.state_dict(), **net.config(), "j": j, "noise_rms": rms}, path + ".tmp")
            os.replace(path + ".tmp", path)
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump({"args": vars(a), "j": j, "coefficients": len(one.plan), "parameters": nparams, "noise_rms": rms,
                   "loss_weights": weights, "seconds": time.time() - started, "loss_history": hist,
                   "evals": evals}, f, indent=1)


def run_joint(a, net, d2, cond, fields, joint_outputs, joint_terms, estimate_bias, weights, evaluate, show, meta,
              transform=None):
    """Adam training of a joint (Q, U) network on the fixed PR3 maps: every step
    denoises both fields at once and sums their terms (``--joint``).  With
    frozen biases, `estimate_bias(z, generator)` re-estimates them every
    `--bias-every` steps at the current output `z` (after `--bias-warmup` steps
    with live biases).
    `transform` (``--pol-stats eb``): the map from the (Q, U) output to the
    fields of the terms."""
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    nparams = sum(q.numel() for q in net.parameters())
    bias_gen = torch.Generator(device=d2.device).manual_seed(a.seed + 777)
    bias = None
    hist = {"step": [], "loss": [], **{t: [] for t in TERMS}}
    evals = {0: evaluate(outputs=joint_outputs())}
    show(0, evals[0])
    started = time.time()
    for step in range(1, a.steps + 1):
        u2 = net(d2[None], d2.std(), cond=cond)[0]
        x = u2.detach().requires_grad_()
        y = x if transform is None else transform(x)  # the fields of the terms
        z = y.detach().requires_grad_()
        if estimate_bias is not None and step > a.bias_warmup and (step - 1 - a.bias_warmup) % a.bias_every == 0:
            bias = estimate_bias(z.detach(), bias_gen)
        terms = {}
        for part in joint_terms(z, bias):
            sum(weights[k] * v for k, v in part.items()).backward()  # gradient into z.grad
            for k, v in part.items():
                terms[k] = terms.get(k, 0) + v.detach()
        loss = sum(weights[k] * v for k, v in terms.items())
        if transform is not None:
            y.backward(z.grad)  # through the transform, into x.grad
        opt.zero_grad(set_to_none=True)
        u2.backward(z.grad if transform is None else x.grad)  # one backward through the network
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
            evals[step] = evaluate(save=step == a.steps, outputs=joint_outputs())
            show(step, evals[step])
            path = os.path.join(a.out, "model.pt")
            torch.save({"state": net.state_dict(), **net.config(), **meta}, path + ".tmp")
            os.replace(path + ".tmp", path)
    with open(os.path.join(a.out, "train.json"), "w") as f:
        json.dump({"args": vars(a), **meta, "parameters": nparams, "loss_weights": weights,
                   "seconds": time.time() - started, "loss_history": hist, "evals": evals}, f, indent=1)


def run_lbfgs(a, fixed, fields, losses, weights, evaluate, show, dev, meta, frozen=None):
    """The reference without a network: minimise the training losses over the
    pixels of u, one map per field, with the native L-BFGS of STL-Dev-MPI
    (strong-Wolfe line search), starting from u = d.

    The loss is the sum over the fixed observations (both fields) of the
    weighted terms of `--loss`, exactly as in training.  Its noise
    realisations are drawn from the training simulations with a seed fixed
    within a round (the loss is then deterministic for the line search) and
    changed between rounds.  No noise bias is frozen (unlike Delouis et al.
    2022): Phi(u + n_i) is recomputed at the current u at every evaluation,
    with the gradient through it.  Every round ends with an
    evaluation; the last one writes eval_{field}.npz.

    With `frozen` (``--bias frozen``: the functions `estimate_bias` and
    `frozen_losses` of :func:`main`), the noise biases are estimated at the
    current u at the start of every round and kept constant during it, and
    the objective is Phi(u) + B: the scheme of Delouis et al. (2022)."""
    from stl_port.native import native_lbfgs

    # The optimiser works on u / std(d), of order one like the input of the network: in uK the
    # first L-BFGS step (lr / |g|_1) is far too short at nside 256.
    scale = np.array([float(fixed[f][1].std()) for f in fields])[:, None]
    x = np.stack([fixed[f][1].detach().cpu().numpy() for f in fields]).astype(np.float64) / scale
    options = {"max_iter": a.lbfgs_iter, "history_size": a.lbfgs_history}
    hist = {"round": [], "evaluations": [], "loss_start": [], "loss_end": [], **{t: [] for t in TERMS}}
    evals = {0: evaluate(outputs={f: fixed[f][1] for f in fields})}
    show(0, evals[0])
    started = time.time()
    for rnd in range(1, a.lbfgs_rounds + 1):
        seed = 1000003 * (a.seed + 1) + rnd
        last = {}
        use_frozen = frozen and rnd > a.bias_warmup
        if use_frozen:
            g = torch.Generator(device=dev).manual_seed(seed)
            bias = {f: frozen[0](torch.as_tensor(x[i] * scale[i], device=dev), f, generator=g)
                    for i, f in enumerate(fields)}

        def objective(trial):
            u = torch.tensor(np.array(trial) * scale, device=dev, requires_grad=True)
            total, parts = 0.0, {}
            for i, f in enumerate(fields):
                _, d, hm = fixed[f]
                if use_frozen:
                    terms = frozen[1](u[i], d, f, bias[f], hm=hm)
                else:
                    g = torch.Generator(device=dev).manual_seed(seed + i)  # the same draws at every evaluation of the round
                    terms = losses(u[i], d, f, generator=g, hm=hm)
                loss_f = sum(weights[k] * v for k, v in terms.items())
                loss_f.backward()  # per field: the engine keeps one forward state at a time
                total += loss_f.item()
                for k, v in terms.items():
                    parts[k] = parts.get(k, 0.0) + v.item()
            last.update(parts)
            return total, u.grad.detach().cpu().numpy() * scale

        x, info = native_lbfgs(objective, x, options)
        h = info["loss_history"]
        hist["round"].append(rnd)
        hist["evaluations"].append(info["evaluations"])
        hist["loss_start"].append(h[0])
        hist["loss_end"].append(min(h))
        for t in TERMS:
            hist[t].append(last.get(t))
        print(f"round {rnd} evaluations {info['evaluations']} loss {h[0]:.6g} -> {min(h):.6g} "
              + " ".join(f"{k} {v:.4g}" for k, v in last.items())
              + f" {(time.time() - started) / rnd:.1f} s/round", flush=True)
        evals[rnd] = evaluate(save=rnd == a.lbfgs_rounds,
                              outputs={f: torch.as_tensor(x[i] * scale[i], device=dev) for i, f in enumerate(fields)})
        show(rnd, evals[rnd])
    with open(os.path.join(a.out, "train.json"), "w") as fh:
        json.dump({"args": vars(a), **meta, "optimizer": "native STL L-BFGS (strong Wolfe)", "loss_weights": weights,
                   "seconds": time.time() - started, "loss_history": hist, "evals": evals}, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
