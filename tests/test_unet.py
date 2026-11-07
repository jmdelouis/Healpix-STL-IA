"""The spherical U-Net: identity at initialisation, gradients, no face edges, patches."""

import math

import healpy as hp
import numpy as np
import pytest
import torch

from stlia import healpix_faces as hf
from stlia.regions import Patches, disc_ids, pixel_vectors
from stlia.unet import UNet, load

NSIDE = 16


def sky(seed=0, nside=NSIDE, batch=2):
    return torch.as_tensor(np.random.default_rng(seed).normal(size=(batch, 12 * nside * nside)))


@pytest.mark.parametrize("conv,norm", [("sphere", "group"), ("sphere", "pixel"), ("faces", "group")])
def test_untrained_network_is_identity(conv, norm):
    net = UNet(NSIDE, levels=3, channels=[4, 8, 8], conv=conv, norm=norm)
    d = sky()
    u = net(d)
    assert u.shape == d.shape and u.dtype == d.dtype
    torch.testing.assert_close(u, d)


def test_gradient_reaches_every_parameter():
    torch.manual_seed(0)
    net = UNet(NSIDE, levels=3, channels=[4, 8, 8])
    torch.nn.init.normal_(net.head.weight, std=0.1)
    net(sky()).square().mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("gauge", ["phi", "cosmo"])
def test_sphere_convolution_has_no_face_edges(gauge):
    """A smooth field through a one-level random network: the finest details
    of the output are no larger along the base-face edges than elsewhere."""
    nside = 32
    torch.manual_seed(0)
    net = UNet(nside, levels=1, channels=[8], gauge=gauge)
    torch.nn.init.normal_(net.head.weight, std=0.1)
    theta, phi = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True)
    m = torch.as_tensor(np.cos(theta) + np.sin(theta) * np.cos(phi))[None]
    with torch.no_grad():
        r = (m - net(m, scale=1.0))[0].numpy()
    det = r - np.repeat(r.reshape(-1, 4).mean(1), 4)
    e = np.zeros((12, nside, nside), bool)
    e[:, 0], e[:, -1], e[:, :, 0], e[:, :, -1] = True, True, True, True
    edge = np.zeros(12 * nside * nside, bool)
    edge[hf.face_nest(nside).ravel()] = e.ravel()
    ratio = np.sqrt((det[edge] ** 2).mean() / (det[~edge] ** 2).mean())
    assert 0.8 < ratio < 1.25


def test_patches_match_full_sky_near_centre():
    """With pixel norm and a fixed scale, the network on a disc equals the
    network on the sky where its receptive field stays inside the disc."""
    nside, levels = 32, 2
    torch.manual_seed(0)
    m = sky(nside=nside, batch=1)
    net = UNet(nside, levels, [8, 8], norm="pixel", scale=1.0)
    torch.nn.init.normal_(net.head.weight, std=0.1)
    vecs = pixel_vectors(nside, "cpu")
    centre, radius = 1000, math.radians(30)
    ids = disc_ids(vecs, centre, radius)
    with torch.no_grad():
        full = net(m)[0]
        part = net.forward_patches(m[:, ids], Patches([ids], nside, levels))[0]
    near = (vecs[ids] @ vecs[centre]) >= math.cos(math.radians(5))
    assert near.sum() > 10
    torch.testing.assert_close(part[near], full[ids][near], atol=5e-4, rtol=1e-4)  # float32 sums in a different order


def test_disc_matches_healpy_query_disc():
    nside = 16
    vecs = pixel_vectors(nside, "cpu")
    ids = disc_ids(vecs, 100, math.radians(20)).numpy()
    ref = np.sort(hp.query_disc(nside, vecs[100].numpy(), math.radians(20), nest=True))
    np.testing.assert_array_equal(ids, ref)


def test_save_and_load(tmp_path):
    net = UNet(NSIDE, levels=2, channels=[4, 8], norm="pixel", scale=3.0)
    torch.nn.init.normal_(net.head.weight, std=0.1)
    torch.save({"state": net.state_dict(), **net.config()}, tmp_path / "model.pt")
    net2, ck = load(tmp_path / "model.pt", "cpu")
    assert ck["scale"] == 3.0
    d = sky()
    with torch.no_grad():
        torch.testing.assert_close(net2(d), net.eval()(d))


def test_conditioning_maps():
    """With a conditioning map: identity at initialisation, the output depends
    on the map once trained, and the patches still match the full sky."""
    nside, levels = 32, 2
    torch.manual_seed(0)
    m = sky(nside=nside, batch=1)
    c = torch.as_tensor(np.random.default_rng(9).normal(size=(1, 1, 12 * nside * nside)))
    net = UNet(nside, levels, [8, 8], norm="pixel", scale=1.0, cond=1)
    torch.testing.assert_close(net(m, cond=c), m)
    torch.nn.init.normal_(net.head.weight, std=0.1)
    with torch.no_grad():
        full = net(m, cond=c)[0]
        assert not torch.allclose(full, net(m, cond=-c)[0])
        vecs = pixel_vectors(nside, "cpu")
        ids = disc_ids(vecs, 1000, math.radians(30))
        part = net.forward_patches(m[:, ids], Patches([ids], nside, levels), cond=c[:, :, ids])[0]
    near = (vecs[ids] @ vecs[1000]) >= math.cos(math.radians(5))
    torch.testing.assert_close(part[near], full[ids][near], atol=5e-4, rtol=1e-4)


def test_joint_qu_network(tmp_path):
    """maps=2: (Q, U) in and out, identity at initialisation, both outputs
    depend on both inputs once trained, and the network saves and loads."""
    torch.manual_seed(0)
    qu = torch.stack([sky(seed=1), sky(seed=2)], 1)  # [B, 2, npix]
    net = UNet(NSIDE, levels=2, channels=[4, 8], maps=2)
    torch.testing.assert_close(net(qu), qu)
    torch.nn.init.normal_(net.head.weight, std=0.1)
    with torch.no_grad():
        u = net(qu, scale=1.0)
        other = qu.clone()
        other[:, 1] *= -1  # change U only
        q_other = net(other, scale=1.0)[:, 0]
    assert u.shape == qu.shape
    assert not torch.allclose(u[:, 0], q_other)  # Q depends on U
    torch.save({"state": net.state_dict(), **net.config()}, tmp_path / "model.pt")
    net2, ck = load(tmp_path / "model.pt", "cpu")
    assert ck["maps"] == 2
    with torch.no_grad():
        torch.testing.assert_close(net2(qu, scale=1.0), net.eval()(qu, scale=1.0))


def test_joint_patches_match_full_sky_near_centre():
    """maps=2 on a disc: [B, 2, n] in and out, equal to the full sky near the centre."""
    nside, levels = 32, 2
    torch.manual_seed(0)
    qu = torch.stack([sky(seed=1, nside=nside, batch=1), sky(seed=2, nside=nside, batch=1)], 1)
    net = UNet(nside, levels, [8, 8], norm="pixel", scale=1.0, maps=2)
    torch.nn.init.normal_(net.head.weight, std=0.1)
    vecs = pixel_vectors(nside, "cpu")
    ids = disc_ids(vecs, 1000, math.radians(30))
    with torch.no_grad():
        full = net(qu)[0]
        part = net.forward_patches(qu[:, :, ids], Patches([ids], nside, levels))[0]
    near = (vecs[ids] @ vecs[1000]) >= math.cos(math.radians(5))
    assert part.shape == (2, len(ids))
    torch.testing.assert_close(part[:, near], full[:, ids][:, near], atol=5e-4, rtol=1e-4)
