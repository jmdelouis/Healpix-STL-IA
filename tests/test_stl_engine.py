"""Statistics of the STL-Dev-MPI engine through torch (skipped without the engine)."""

import numpy as np
import pytest
import torch

native = pytest.importorskip("stl_port.native", reason="STL-Dev-MPI engine not installed")
pytestmark = pytest.mark.stl


def test_autograd_matches_finite_differences():
    from stlia.stl_torch import Stats

    try:
        eng = native.NativeEngine(8, 2, batch=1, backend="cpu")
    except Exception as exc:  # the library is not found
        pytest.skip(f"engine unavailable: {exc}")
    stats = Stats(eng)
    x = torch.tensor(np.random.default_rng(0).normal(size=(1, 768)), requires_grad=True)
    w = torch.as_tensor(np.random.default_rng(1).normal(size=(len(eng.plan), 2)))

    def f(y):
        return (stats(y)[0] * w).sum()

    f(x).backward()
    dx = torch.zeros_like(x)
    dx[0, 5] = 1e-6
    fd = (f(x.detach() + dx) - f(x.detach() - dx)) / 2e-6
    torch.testing.assert_close(x.grad[0, 5], fd, rtol=1e-4, atol=1e-8)
