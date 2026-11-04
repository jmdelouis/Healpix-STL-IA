"""Checks of the sphere convolution of `stlia.unet` (run on a GPU node).

    python scripts/check_sphere_conv.py [--nside 64]

1. the stencil weights of every pixel and tap sum to 1 (bilinear interpolation);
2. a constant map is left constant by a kernel that sums to one;
3. gradient of the convolution against finite differences (float64);
4. no face edges: the response of a random kernel to a smooth field (a
   low-order spherical harmonic) has no excess near the base-face edges,
   compared with the face-image convolution;
5. time of a forward + backward pass of the U-Net.
"""

import argparse
import time

import numpy as np
import torch

from stlia import healpix_faces as hf
from stlia.unet import SphereConv, SphereLevel, UNet


def edge_mask(nside, width=1):
    """NESTED pixels within `width` pixels of a base-face edge."""
    f = hf.face_nest(nside)
    m = np.zeros((12, nside, nside), bool)
    m[:, :width], m[:, -width:], m[:, :, :width], m[:, :, -width:] = True, True, True, True
    out = np.zeros(12 * nside * nside, bool)
    out[f.ravel()] = m.ravel()
    return out


def smooth_field(nside):
    import healpy as hp

    theta, phi = hp.pix2ang(nside, np.arange(12 * nside * nside), nest=True)
    return np.sin(theta) ** 2 * np.cos(2 * phi) + 0.5 * np.cos(theta) ** 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=64)
    a = p.parse_args()
    dev = torch.device("cuda")
    n = a.nside

    for gauge in ("phi", "cosmo"):
        lv = SphereLevel(n, 3, gauge, dev)
        rows = torch.sparse.mm(lv.S, torch.ones(lv.npix, 1, device=dev))
        print(f"[{gauge}] stencil row sums: min {rows.min().item():.6f} max {rows.max().item():.6f}")

    lv = SphereLevel(n, 3, "phi", dev)
    conv = SphereConv(1, 1).to(dev)
    with torch.no_grad():
        conv.weight.fill_(1.0 / 9)
        conv.bias.zero_()
        y = conv(torch.full((1, 1, lv.npix), 2.5, device=dev), lv)
    print(f"constant map: output in [{y.min().item():.6f}, {y.max().item():.6f}] (expected 2.5)")

    small = SphereLevel(4, 3, "phi", dev)
    S64, St64 = small.S.to(torch.float64), small.St.to(torch.float64)
    small.S, small.St = S64, St64
    c2 = SphereConv(2, 3).to(dev, torch.float64)
    x = torch.randn(2, 2, small.npix, device=dev, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda t: c2(t, small), (x,), eps=1e-6, atol=1e-6)
    print(f"gradcheck (nside 4, float64): {ok}")

    s = torch.as_tensor(smooth_field(n), device=dev, dtype=torch.float32)
    edge = edge_mask(n)
    torch.manual_seed(0)
    for conv_type in ("sphere", "faces"):
        net = UNet(n, levels=1, channels=[8], conv=conv_type).to(dev)
        with torch.no_grad():
            torch.nn.init.normal_(net.head.weight)
            r = (s[None] - net(s[None].double(), torch.tensor(1.0, device=dev).double()).float())[0].cpu().numpy()
        lap = r - np.repeat(r.reshape(-1, 4).mean(1), 4)
        print(f"[{conv_type}] one-level net on a smooth field: rms of the 4-children detail "
              f"at face edges / elsewhere = {np.sqrt((lap[edge] ** 2).mean()) / np.sqrt((lap[~edge] ** 2).mean()):.3f}")

    for conv_type in ("sphere", "faces"):
        net = UNet(n, conv=conv_type).to(dev)
        d = torch.randn(1, 12 * n * n, device=dev, dtype=torch.float64)
        for i in range(6):
            if i == 1:
                torch.cuda.synchronize()
                t0 = time.time()
            net(d, d.std()).sum().backward()
        torch.cuda.synchronize()
        print(f"[{conv_type}] U-Net forward + backward at nside {n}: {(time.time() - t0) / 5 * 1e3:.1f} ms, "
              f"{sum(q.numel() for q in net.parameters())} parameters")


if __name__ == "__main__":
    main()
