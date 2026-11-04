"""HEALPix geometry of the torch port against healpy."""

import healpy as hp
import numpy as np
import pytest
import torch

from stlia import healpix_faces as hf
from stlia import stl_torch_geometry as tg


@pytest.mark.parametrize("order", [0, 2, 4])
def test_nest_xyf_roundtrip(order):
    pix = torch.arange(12 * 4**order)
    ix, iy, face = tg.nest2xyf(pix, order)
    assert torch.equal(tg.xyf2nest(ix, iy, face, order), pix)


@pytest.mark.parametrize("nside", [1, 4, 16])
def test_pixel_centres_match_healpy(nside):
    order = int(np.log2(nside))
    pix = torch.arange(12 * nside * nside)
    theta, phi = tg.xyf2ang(*tg.nest2xyf(pix, order), nside)
    t_ref, p_ref = hp.pix2ang(nside, pix.numpy(), nest=True)
    np.testing.assert_allclose(theta.double().numpy(), t_ref, atol=1e-6)
    dphi = np.angle(np.exp(1j * (phi.double().numpy() - p_ref)))
    np.testing.assert_allclose(dphi, 0, atol=1e-6)


@pytest.mark.parametrize("nside", [4, 16])
def test_ring2nest_matches_healpy(nside):
    order = int(np.log2(nside))
    ring = torch.arange(12 * nside * nside)
    np.testing.assert_array_equal(tg.ring2nest(ring, order).numpy(), hp.ring2nest(nside, ring.numpy()))


def test_interpolation_reproduces_healpy():
    nside, order = 16, 4
    rng = np.random.default_rng(0)
    vec = rng.normal(size=(200, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    nb, w = tg.interp_weights_nested(torch.as_tensor(vec), order)
    np.testing.assert_allclose(w.sum(-1).numpy(), 1.0, atol=1e-12)
    theta, phi = hp.vec2ang(vec)
    p_ref, w_ref = hp.get_interp_weights(nside, theta, phi, nest=True)
    m = rng.normal(size=12 * nside * nside)
    np.testing.assert_allclose((m[nb.numpy()] * w.numpy()).sum(-1), (m[p_ref] * w_ref).sum(0), atol=1e-10)


def test_faces_roundtrip_and_pyramid():
    nside = 8
    m = np.random.default_rng(1).normal(size=12 * nside * nside)
    np.testing.assert_array_equal(hf.from_faces(hf.to_faces(m)), m)
    np.testing.assert_allclose(hf.down(m), m.reshape(-1, 4).mean(1))
    np.testing.assert_array_equal(hf.up(hf.down(m)), np.repeat(hf.down(m), 4))
