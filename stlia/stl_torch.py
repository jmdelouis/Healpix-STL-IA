"""The Rust scattering-covariance engine of STL-Dev-MPI as a PyTorch layer.

(`StatsFunction` and `Stats` from STL-Dev-MPI `amortized/stl_torch.py`.)

`StatsFunction` wraps `NativeEngine.forward` / `.vjp` of `stl-bridge`: the
statistics of a batch of NESTED maps `[B, 12 nside^2]` (float64) as a real
tensor `[B, ncoeff, 2]` (real and imaginary parts), and in the backward pass
the exact map cotangent from the pyramidal reverse sweep
(`cotangent = dL/dRe + i dL/dIm`).

`DenoiseLoss` is the data term of the scattering denoising of Delouis et al.
(2022): with `u` the denoised map and `n_i` white-noise realisations,

    loss = sum_c w_c |mean_i S_c(u + n_i) - S_c(d)|^2,

    w_c = 1 / (n sigma_c^2),   sigma_c^2 = V_c(d) / N_c,

`V_c(d)` the spatial variance of the terms of coefficient `c` on `d` over the
`N_c` pixels of its native level (the ergodic weights of `stl-synth`), `n` the
number of coefficients: a reduced chi^2 of the mean over the realisations.
"""

import numpy as np
import torch

FAMILIES = ("mean", "var", "S1", "S2", "S3", "S4")


class StatsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, engine, generation):
        values, _ = engine.forward(x.detach().cpu().numpy(), generation=generation)
        ctx.engine, ctx.generation, ctx.device = engine, generation, x.device
        out = values.view(np.float64).reshape(values.shape + (2,))
        return torch.from_numpy(out.copy()).to(x.device)

    @staticmethod
    def backward(ctx, g):
        g = g.detach().cpu().numpy()
        grad = ctx.engine.vjp(g[..., 0] + 1j * g[..., 1], generation=ctx.generation)
        return torch.from_numpy(grad).to(ctx.device), None, None


class Stats:
    """Statistics of a batch with gradient; one engine, one forward state at a time."""

    def __init__(self, engine):
        self.engine = engine
        self.generation = 0

    def __call__(self, x):
        self.generation += 1
        return StatsFunction.apply(x, self.engine, self.generation)


def native_pixels(plan, nside):
    return np.array([12 * (nside >> c["native_dg"]) ** 2 for c in plan], dtype=np.float64)


def as_real(values):
    """Complex `[..., ncoeff]` -> real `[..., ncoeff, 2]`."""
    return np.stack([values.real, values.imag], -1)


class DenoiseLoss:
    def __init__(self, plan, nside, device="cpu"):
        self.n = len(plan)
        self.npix = torch.as_tensor(native_pixels(plan, nside), device=device)
        self.family = np.array([c["family"] for c in plan])

    def weights(self, target_var):
        """`target_var` `[ncoeff]` (moments of the forward of `d`) -> (w, sigma^2)."""
        sigma2 = torch.clamp(target_var / self.npix, min=1e-300)
        w = torch.where(target_var > 0, 1.0 / sigma2, torch.zeros_like(sigma2))
        w = torch.where(target_var > 0, w, w.max())
        return w / self.n, sigma2

    def __call__(self, s, target, target_var):
        """`s` `[B, ncoeff, 2]` statistics of `u + n_i`; `target` `[ncoeff, 2]`
        and `target_var` `[ncoeff]` those of `d`."""
        w, _ = self.weights(target_var)
        return (w * ((s.mean(0) - target) ** 2).sum(-1)).sum()

    def report(self, s, target, target_var):
        """Per family: rms of `(s - target) / sigma` (numpy or torch, `s` `[ncoeff, 2]`)."""
        s, t, v = (torch.as_tensor(a, device=self.npix.device) for a in (s, target, target_var))
        _, sigma2 = self.weights(v)
        z2 = (((s - t) ** 2).sum(-1) / sigma2).cpu().numpy()
        return {f: float(np.sqrt(z2[self.family == f].mean())) for f in FAMILIES if (self.family == f).any()}
