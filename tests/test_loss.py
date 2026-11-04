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
