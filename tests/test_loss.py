"""Reduced chi^2 with ergodic weights (no engine needed: a synthetic plan)."""

import numpy as np
import torch

from stlia.stl_torch import DenoiseLoss, as_real, native_pixels


def plan(n=6):
    fam = ["mean", "var", "S1", "S1", "S2", "S3"][:n]
    return [{"family": f, "native_dg": k % 3} for k, f in enumerate(fam)]


def test_as_real():
    v = np.array([1 + 2j, 3 - 4j])
    np.testing.assert_array_equal(as_real(v), [[1, 2], [3, -4]])


def test_native_pixels():
    np.testing.assert_array_equal(native_pixels(plan(3), 8), [768, 192, 48])


def test_loss_is_a_reduced_chi2():
    p = plan()
    loss = DenoiseLoss(p, 8)
    target = torch.zeros(len(p), 2, dtype=torch.float64)
    var = torch.ones(len(p), dtype=torch.float64)
    sigma = torch.sqrt(var / loss.npix)
    s = torch.zeros(3, len(p), 2, dtype=torch.float64)
    s[..., 0] = sigma  # every coefficient one ergodic sigma off
    torch.testing.assert_close(loss(s, target, var), torch.tensor(1.0, dtype=torch.float64))
    rep = loss.report(s[0], target, var)
    assert set(rep) == {"mean", "var", "S1", "S2", "S3"}
    np.testing.assert_allclose(list(rep.values()), 1.0)


def test_loss_with_masks():
    """K masks: coefficients flattened mask-major, one ergodic weight each."""
    p = plan()
    npix = np.array([[768.0] * len(p), [300.0] * len(p)])
    loss = DenoiseLoss(p, 8, masks=2, npix=npix)
    assert loss.n == 2 * len(p) and loss.npix.shape == (2 * len(p),)
    target = torch.zeros(2 * len(p), 2, dtype=torch.float64)
    var = torch.ones(2 * len(p), dtype=torch.float64)
    s = torch.zeros(1, 2 * len(p), 2, dtype=torch.float64)
    s[..., 0] = torch.sqrt(var / loss.npix)
    torch.testing.assert_close(loss(s, target, var), torch.tensor(1.0, dtype=torch.float64))


def test_unbiased_coefficients():
    """The noise bias is zero only where the noise enters linearly."""
    from stlia.train_denoise import unbiased
    plan = [{"family": f, "c1": c1} for f, c1 in
            (("mean", -1), ("var", -1), ("S1", 0), ("S2", 0), ("S3", 0), ("S3", 1), ("S4", 0), ("S4", 1))]
    #                           mean   var    S1     S2     S3 |a| S3 |b| S4     S4
    assert list(unbiased(plan, "data")) == [1, 0, 0, 0, 0, 0, 0, 0]
    assert list(unbiased(plan, "l1")) == [0, 0, 0, 1, 0, 0, 0, 0]
    assert list(unbiased(plan, "l2")) == [0, 0, 0, 1, 0, 1, 0, 0]  # Phi(u + n, u): modulus of u
    assert list(unbiased(plan, "l3")) == [0, 0, 0, 1, 1, 0, 0, 0]  # Phi(T, u + n): modulus of T
