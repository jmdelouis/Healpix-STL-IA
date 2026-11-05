"""Binned, masked angular power spectra as a differentiable loss.

The loss ``cl`` asks the denoised map to have the power spectrum of the sky.
Its noise-free target is the half-mission cross spectrum ``C_b(d1, d2)``, and
its model the spectrum of ``u``:

    cl    C_b(m d1, m d2)  vs  C_b(m u, m u)

since the two half-mission noises are independent, ``< C_b(u + n1, u + n2) >``
equals ``C_b(u, u)``: the model needs no noise draw and no noise bias,
for every galactic mask ``m`` (``|b|`` above a cut, apodised).  The spectra are
pseudo-spectra of the masked maps divided by the mean of the squared mask,
binned in multipole, and compared with a Gaussian estimate of the variance
of each bin of the target.

Two further terms use the same spectra.  ``tcl`` compares the cross spectrum
of a noise-free template ``T`` (the 857 GHz map) with ``u`` to that with
``d``: the noise is independent of ``T``, so ``< C_b(T, d) > = C_b(T, s)``.
``rcl`` asks the removed part ``d - u`` to have the spectrum of the noise
(:class:`NoiseSpectrumLoss`).

With ``spin=0`` the spectra are those of scalar maps (Q and U separately, as
the network treats them by default).  With ``spin=2`` the maps are (Q, U)
pairs and the spectra are EE and BB, for a joint (Q, U) network
(``UNet(maps=2)``); they need the ``stl`` backend.

Two backends compute the pseudo-spectra: ``torch``, the ring-FFT transform of
:mod:`stlia.sht` through ``torch.autograd``, and ``stl``, the differentiable
``anafast`` of the STL-Dev-MPI engine (``stl_port.native.SpectrumEngine``,
branch ``claude/spectrum``), on the CPU or the GPU.
"""

import math

import healpy as hp
import numpy as np
import torch

from .sht import HEALPixSHT


class _EngineCross(torch.autograd.Function):
    """Pseudo cross spectra `[B, K, lmax + 1]` of maps `a`, `b` `[B, npix]`
    (spin 2: `[B, K, 4, lmax + 1]` of (Q, U) pairs `[B, 2, npix]`, components
    E1E2, B1B2, E1B2, B1E2) by the STL-Dev-MPI engine, with its VJP."""

    @staticmethod
    def forward(ctx, a, b, native):
        engine = native.engine(a.shape[0])
        x = torch.stack([a, b], 1).detach().cpu().double().numpy()
        native.generation += 1
        key = native.generation
        cl = engine.forward(x, generation=key)
        ctx.native, ctx.key, ctx.x = native, key, x
        return torch.as_tensor(cl, dtype=a.dtype, device=a.device)

    @staticmethod
    def backward(ctx, cot):
        engine = ctx.native.engine(ctx.x.shape[0])
        c = cot.detach().cpu().double().numpy()
        try:
            g = engine.vjp(c, generation=ctx.key)
        except RuntimeError:  # another forward ran since: redo this one
            engine.forward(ctx.x, generation=ctx.key)
            g = engine.vjp(c, generation=ctx.key)
        g = torch.as_tensor(g, dtype=cot.dtype, device=cot.device)
        return g[:, 0], g[:, 1], None


class _Native:
    """The cross-spectrum engines of STL-Dev-MPI, one per batch size."""

    def __init__(self, nside, lmax, masks, device, spin=0):
        self.nside, self.lmax, self.masks, self.spin = nside, lmax, masks, spin
        dev = torch.device(device)
        self.backend = "gpu" if dev.type == "cuda" else "cpu"
        self.index = dev.index or 0
        self.engines, self.generation = {}, 0

    def engine(self, batch):
        if batch not in self.engines:
            from stl_port.native import SpectrumEngine
            self.engines[batch] = SpectrumEngine(self.nside, lmax=self.lmax, batch=batch, backend=self.backend,
                                                 device=self.index, cross=True, masks=self.masks,
                                                 **({"spin": self.spin} if self.spin else {}))
        return self.engines[batch]


def galactic_mask(nside, bcut, apod=5.0):
    """NESTED mask `[npix]` of ``|b| > bcut`` degrees, smoothed by a Gaussian of
    `apod` degrees FWHM and clipped to [0, 1]; `bcut` = 0 is the full sky."""
    if bcut <= 0:
        return np.ones(12 * nside * nside)
    theta, _ = hp.pix2ang(nside, np.arange(12 * nside * nside))
    m = (np.abs(90.0 - np.degrees(theta)) > bcut).astype(float)
    if apod > 0:
        m = np.clip(hp.smoothing(m, fwhm=math.radians(apod)), 0.0, 1.0)
    return hp.reorder(m, r2n=True)


def bin_edges(lmin, lmax, width):
    """Multipole bins [lmin, lmin + width), ..., ending at lmax (inclusive)."""
    edges = list(range(lmin, lmax + 1, width))
    if edges[-1] != lmax + 1:
        edges.append(lmax + 1)
    return np.array(edges)


class BinnedSpectra:
    """Binned pseudo-spectra of NESTED maps under several masks.

    Parameters
    ----------
    nside : int
        Resolution of the maps.
    masks : list of float
        Galactic cuts in degrees (0: full sky).
    lmin, lmax, width : int
        Multipole bins.
    apod : float
        FWHM of the mask apodisation, degrees.
    device : str or torch.device
    backend : {"torch", "stl"}
        ``torch``: the transforms of :mod:`stlia.sht`; ``stl``: the
        ``SpectrumEngine`` of STL-Dev-MPI, on the GPU when `device` is one
        (the engine is built at the first call of each batch size).
    spin : {0, 2}
        0: spectra of scalar maps `[..., npix]`; 2: EE and BB spectra of
        (Q, U) pairs `[..., 2, npix]`, ``stl`` backend only.
    """

    def __init__(self, nside, masks=(0.0, 30.0), lmin=10, lmax=None, width=30, apod=5.0, device="cpu",
                 backend="torch", spin=0):
        self.lmax = lmax or 2 * nside - 1
        self.cuts = list(masks)
        m = np.stack([galactic_mask(nside, b, apod) for b in self.cuts])
        assert spin in (0, 2), "spin 0 or 2"
        assert spin == 0 or backend == "stl", "EE/BB spectra: stl backend only"
        self.spin = spin
        if backend == "torch":
            self.sht = HEALPixSHT(nside, lmax=self.lmax, dtype=torch.float64, device=device)
        elif backend == "stl":
            self.native = _Native(nside, self.lmax, m, device, spin)
        else:
            raise ValueError("backend must be torch or stl")
        self.backend = backend
        self.masks = torch.as_tensor(m, device=device)
        self.w2 = self.masks.square().mean(1)  # [M]
        self.edges = bin_edges(lmin, self.lmax, width)
        nb = len(self.edges) - 1
        ell = np.arange(self.lmax + 1)
        b = np.zeros((nb, self.lmax + 1))
        for i, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            b[i, lo:hi] = 1.0 / (hi - lo)
        self.binning = torch.as_tensor(b, device=device)  # [nb, lmax + 1]
        # modes per bin: sum (2l + 1) times the effective sky fraction of each mask
        modes = np.array([(2 * ell[lo:hi] + 1).sum() for lo, hi in zip(self.edges[:-1], self.edges[1:])])
        fsky = (self.w2 ** 2 / self.masks.pow(4).mean(1)).cpu().numpy()  # [M]
        self.modes = torch.as_tensor(fsky[:, None] * modes[None], device=device)  # [M, nb]

    def __call__(self, a, b=None):
        """Binned cross spectra `[..., M, nb]` of maps `a` (and `b`) `[..., npix]`;
        spin 2: `[..., M, 2, nb]` (EE, BB) of (Q, U) pairs `[..., 2, npix]`."""
        if self.backend == "stl":
            b = a if b is None else b
            field = a.shape[-1:] if self.spin == 0 else a.shape[-2:]
            lead = a.shape[:len(a.shape) - len(field)]
            cl = _EngineCross.apply(a.reshape(-1, *field), b.reshape(-1, *field), self.native)
            w2 = self.w2[:, None].to(cl.dtype)
            if self.spin:
                cl, w2 = cl[:, :, :2], w2[:, None]  # E1E2, B1B2
            cl = cl.reshape(*lead, *cl.shape[1:]) / w2
            return cl @ self.binning.T.to(cl.dtype)
        ma = a[..., None, :] * self.masks
        mb = ma if b is None else b[..., None, :] * self.masks
        cl = self.sht.anafast(ma, mb, nest=True) / self.w2[:, None]  # [..., M, lmax + 1]
        return cl @ self.binning.T

    def variance(self, c11, c22, c12):
        """Gaussian variance of a binned cross spectrum from the auto and cross
        spectra of the two maps: (C11 C22 + C12^2) / modes."""
        modes = self.modes[:, None] if self.spin else self.modes
        return (c11 * c22 + c12 ** 2) / modes


class SpectrumLoss:
    """Reduced chi^2 between binned spectra of `u` and the half-mission cross
    spectrum of the data (see the module docstring)."""

    def __init__(self, spectra, d1, d2):
        self.spectra = spectra
        with torch.no_grad():
            self.target = spectra(d1, d2)
            self.var = spectra.variance(spectra(d1), spectra(d2), self.target).clamp_min(1e-300)

    def __call__(self, model):
        """`model` shaped like the target (or with a leading batch, averaged) -> scalar."""
        if model.dim() > self.target.dim():
            model = model.mean(0)
        return ((model - self.target) ** 2 / self.var).mean()


class NoiseSpectrumLoss(SpectrumLoss):
    """Reduced chi^2 between the binned spectrum of the removed part `d - u`
    and the mean spectrum of noise maps `[k, npix]` (simulations, or the
    half-mission difference of the data): the term ``rcl``,

        rcl   < C_b(m n, m n) >  vs  C_b(m (d - u), m (d - u))

    which a perfect denoising satisfies at every scale.  The variance is the
    Gaussian one of a single realisation, 2 C_b^2 / modes."""

    def __init__(self, spectra, noises, chunk=8):
        self.spectra = spectra
        with torch.no_grad():
            cl = torch.cat([spectra(noises[i:i + chunk]) for i in range(0, len(noises), chunk)])
            self.target = cl.mean(0)
            self.var = spectra.variance(self.target, self.target, self.target).clamp_min(1e-300)
