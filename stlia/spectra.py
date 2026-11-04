"""Binned, masked angular power spectra as a differentiable loss.

The loss ``cl`` asks the denoised map to have the power spectrum of the sky.
Its noise-free target is the half-mission cross spectrum ``C_b(d1, d2)``, and
its model the same estimator on ``u`` plus fresh half-mission noise draws:

    cl    C_b(m d1, m d2)  vs  < C_b(m (u + n1), m (u + n2)) >

for every galactic mask ``m`` (``|b|`` above a cut, apodised).  The spectra are
pseudo-spectra of the masked maps divided by the mean of the squared mask,
binned in multipole, and compared with a Gaussian estimate of the variance
of each bin of the target.

The spectra are those of the scalar maps (Q and U separately, as the network
treats them); E/B spectra need a joint (Q, U) network.
"""

import math

import healpy as hp
import numpy as np
import torch

from .sht import HEALPixSHT


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
    """

    def __init__(self, nside, masks=(0.0, 30.0), lmin=10, lmax=None, width=30, apod=5.0, device="cpu"):
        self.lmax = lmax or 2 * nside - 1
        self.sht = HEALPixSHT(nside, lmax=self.lmax, dtype=torch.float64, device=device)
        self.cuts = list(masks)
        m = np.stack([galactic_mask(nside, b, apod) for b in self.cuts])
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
        """Binned cross spectra `[..., M, nb]` of maps `a` (and `b`) `[..., npix]`."""
        ma = a[..., None, :] * self.masks
        mb = ma if b is None else b[..., None, :] * self.masks
        cl = self.sht.anafast(ma, mb, nest=True) / self.w2[:, None]  # [..., M, lmax + 1]
        return cl @ self.binning.T

    def variance(self, c11, c22, c12):
        """Gaussian variance of a binned cross spectrum from the auto and cross
        spectra of the two maps: (C11 C22 + C12^2) / modes."""
        return (c11 * c22 + c12 ** 2) / self.modes


class SpectrumLoss:
    """Reduced chi^2 between binned spectra of `u` and the half-mission cross
    spectrum of the data (see the module docstring)."""

    def __init__(self, spectra, d1, d2):
        self.spectra = spectra
        with torch.no_grad():
            self.target = spectra(d1, d2)
            self.var = spectra.variance(spectra(d1), spectra(d2), self.target).clamp_min(1e-300)

    def __call__(self, model):
        """`model` `[M, nb]` (or `[B, M, nb]`, averaged over B) -> scalar."""
        if model.dim() == 3:
            model = model.mean(0)
        return ((model - self.target) ** 2 / self.var).mean()
