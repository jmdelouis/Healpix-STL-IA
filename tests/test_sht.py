"""Differentiable spherical harmonic transforms and spectra against healpy."""

import healpy as hp
import numpy as np
import pytest
import torch

from stlia.sht import HEALPixSHT
from stlia.spectra import BinnedSpectra, SpectrumLoss, galactic_mask

NSIDE = 16


def nest_map(seed):
    rng = np.random.default_rng(seed)
    cl = 1.0 / (1.0 + np.arange(3 * NSIDE)) ** 2
    m = hp.synfast(cl, NSIDE) + 0.1 * rng.normal(size=12 * NSIDE * NSIDE)
    return hp.reorder(m, r2n=True)


def test_anafast_matches_healpy():
    sht = HEALPixSHT(NSIDE, dtype=torch.float64, device="cpu")
    m1, m2 = nest_map(0), nest_map(1)
    cl = sht.anafast(torch.as_tensor(m1), torch.as_tensor(m2), nest=True).numpy()
    ref = hp.anafast(hp.reorder(m1, n2r=True), hp.reorder(m2, n2r=True), lmax=3 * NSIDE - 1, iter=0)
    np.testing.assert_allclose(cl, ref, rtol=1e-6, atol=1e-10 * np.abs(ref).max())


def test_map2alm_matches_healpy():
    sht = HEALPixSHT(NSIDE, dtype=torch.float64, device="cpu")
    m = nest_map(2)
    alm = sht.map2alm(torch.as_tensor(m), nest=True).numpy()
    ref = hp.map2alm(hp.reorder(m, n2r=True), lmax=3 * NSIDE - 1, iter=0)
    np.testing.assert_allclose(alm, ref, atol=1e-8 * np.abs(ref).max())


def test_spectrum_gradient():
    sht = HEALPixSHT(NSIDE, lmax=2 * NSIDE - 1, dtype=torch.float64, device="cpu")
    x = torch.tensor(nest_map(3), requires_grad=True)
    w = torch.as_tensor(np.random.default_rng(4).normal(size=2 * NSIDE))
    f = lambda y: (sht.anafast(y, nest=True) * w).sum()  # noqa: E731
    f(x).backward()
    dx = torch.zeros_like(x)
    dx[17] = 1e-6
    fd = (f(x.detach() + dx) - f(x.detach() - dx)) / 2e-6
    torch.testing.assert_close(x.grad[17], fd, rtol=1e-5, atol=1e-12)


def test_galactic_mask():
    m = galactic_mask(NSIDE, 30.0, apod=0.0)
    theta, _ = hp.pix2ang(NSIDE, np.arange(m.size), nest=True)
    np.testing.assert_array_equal(m > 0.5, np.abs(90 - np.degrees(theta)) > 30)
    assert galactic_mask(NSIDE, 0.0).min() == 1.0


def test_spectrum_loss_zero_at_target():
    spec = BinnedSpectra(NSIDE, masks=(0.0, 20.0), lmin=2, width=4, apod=10.0)
    d = torch.as_tensor(nest_map(5))
    loss = SpectrumLoss(spec, d, d)
    assert float(loss(spec(d, d))) < 1e-20
    assert float(loss(spec(1.2 * d, 1.2 * d))) > 0


def test_engine_spectra_match_torch():
    native = pytest.importorskip("stl_port.native")
    if not hasattr(native, "SpectrumEngine"):
        pytest.skip("STL-Dev-MPI without SpectrumEngine")
    kw = dict(masks=(0.0, 20.0), lmin=2, width=4, apod=10.0)
    ref, eng = BinnedSpectra(NSIDE, **kw), BinnedSpectra(NSIDE, backend="stl", **kw)
    a = torch.as_tensor(np.stack([nest_map(6), nest_map(7)]))
    b = torch.as_tensor(np.stack([nest_map(8), nest_map(9)]))
    torch.testing.assert_close(eng(a, b), ref(a, b), rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(eng(a[0]), ref(a[0]), rtol=1e-8, atol=1e-12)
    w = torch.as_tensor(np.random.default_rng(10).normal(size=ref(a, b).shape))
    grads = []
    for spec in (ref, eng):
        x, y = a.clone().requires_grad_(), b.clone().requires_grad_()
        (spec(x, y) * w).sum().backward()
        grads.append((x.grad, y.grad))
    for g, h in zip(*grads):
        torch.testing.assert_close(g, h, rtol=1e-7, atol=1e-12 * float(h.abs().max()))
