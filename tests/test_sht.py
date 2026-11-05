"""Differentiable spherical harmonic transforms and spectra against healpy."""

import healpy as hp
import numpy as np
import pytest
import torch

from stlia.sht import HEALPixSHT
from stlia.spectra import BinnedSpectra, NoiseSpectrumLoss, SpectrumLoss, galactic_mask

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


def test_noise_spectrum_loss():
    # rcl: a residual drawn like the noise scores about 1, a residual of zero far more
    spec = BinnedSpectra(NSIDE, masks=(0.0, 20.0), lmin=2, width=4, apod=10.0)
    noise = torch.as_tensor(np.random.default_rng(6).normal(size=(41, 12 * NSIDE * NSIDE)))
    loss = NoiseSpectrumLoss(spec, noise[:40])
    assert 0.2 < float(loss(spec(noise[40]))) < 5
    assert float(loss(spec(0.0 * noise[40]))) > 20


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


def test_engine_ee_bb_match_healpy():
    native = pytest.importorskip("stl_port.native")
    if not hasattr(native, "SpectrumEngine") or "spin" not in native.SpectrumEngine.__init__.__code__.co_varnames:
        pytest.skip("STL-Dev-MPI without spin spectra")
    cuts, lmin, width, apod = (0.0, 20.0), 2, 4, 10.0
    spec = BinnedSpectra(NSIDE, masks=cuts, lmin=lmin, width=width, apod=apod, backend="stl", spin=2)
    a = np.stack([nest_map(11), nest_map(12)])  # (Q, U) NESTED
    b = np.stack([nest_map(13), nest_map(14)])
    got = spec(torch.as_tensor(a), torch.as_tensor(b)).numpy()  # [M, 2, nb]
    lmax = 2 * NSIDE - 1
    for k, cut in enumerate(cuts):
        w = hp.reorder(galactic_mask(NSIDE, cut, apod), n2r=True)
        ring = lambda m: [np.zeros_like(w)] + [w * hp.reorder(x, n2r=True) for x in m]  # noqa: E731
        cl = hp.anafast(ring(a), ring(b), pol=True, lmax=lmax, iter=0)[1:3] / np.mean(w ** 2)
        ref = cl @ spec.binning.numpy().T
        np.testing.assert_allclose(got[k], ref, rtol=1e-8, atol=1e-10 * np.abs(ref).max())


def test_eb_maps_match_healpy():
    from stlia.polar import EBTransform
    q, u = nest_map(5), nest_map(6)
    eb = EBTransform(NSIDE, device="cpu", dtype=torch.float64)
    qu = torch.tensor(np.stack([q, u]), requires_grad=True)
    out = eb(qu)
    alm = hp.map2alm([np.zeros_like(q), hp.reorder(q, n2r=True), hp.reorder(u, n2r=True)],
                     lmax=3 * NSIDE - 1, iter=0, pol=True)
    for k in (0, 1):
        ref = hp.reorder(hp.alm2map(alm[k + 1], NSIDE, lmax=3 * NSIDE - 1), r2n=True)
        np.testing.assert_allclose(out[k].detach().numpy(), ref, atol=1e-8 * np.abs(ref).max())
    out.square().sum().backward()  # differentiable with respect to (Q, U)
    assert torch.isfinite(qu.grad).all() and qu.grad.abs().sum() > 0


def test_noise_spectrum_from_halves():
    # C(d) - C(d1, d2) is the spectrum of the noise of d: d - u = n scores about 1
    spec = BinnedSpectra(NSIDE, masks=(0.0,), lmin=2, width=4, apod=0.0)
    rng = np.random.default_rng(7)
    s = torch.as_tensor(nest_map(8))
    n1, n2 = (torch.as_tensor(0.3 * rng.normal(size=s.numel())) for _ in range(2))
    d = s + 0.5 * (n1 + n2)
    loss = NoiseSpectrumLoss.from_halves(spec, d, s + n1, s + n2)
    assert float(loss(spec(d - s))) < 3
    assert float(loss(spec(0.0 * d))) > 20
