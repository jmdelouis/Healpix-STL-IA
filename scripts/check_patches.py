"""Consistency of the U-Net on disc patches with the U-Net on the full sky.

    python scripts/check_patches.py MAP.npy [--radius 10 --levels 3 4 --patches 8]

A network with random weights (pixel norm, fixed input scale, non-zero head)
is applied to the full NESTED map and to disc patches of `radius` degrees
around random centres.  The difference between the two outputs, relative to
the rms of the full-sky correction, is reported per ring of distance to the
centre: it is zero where the receptive field of the network stays inside the
patch, and sets the margin (halo) to leave between the patch edge and the
region where the loss is computed.  Also: time of the geometry and of a
forward + backward pass on the batch of patches against the full sky.
"""

import argparse
import math
import time

import numpy as np
import torch

from stlia.regions import Patches, disc_ids, pixel_vectors
from stlia.unet import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("map")
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--levels", type=int, nargs="+", default=[3, 4])
    p.add_argument("--patches", type=int, default=8)
    p.add_argument("--channels", default="16,32,48,64,64,64")
    a = p.parse_args()
    dev = torch.device("cuda")
    m = torch.as_tensor(np.load(a.map), device=dev)
    nside = int(round(math.sqrt(m.numel() / 12)))
    vecs = pixel_vectors(nside, dev)
    g = torch.Generator(device="cpu").manual_seed(1)
    centres = torch.randint(0, m.numel(), (a.patches,), generator=g).tolist()
    ids = [disc_ids(vecs, c, math.radians(a.radius)) for c in centres]
    print(f"nside {nside}, {a.patches} discs of {a.radius} deg: {sum(len(i) for i in ids)} pixels "
          f"({sum(len(i) for i in ids) / m.numel():.3f} of the sky)", flush=True)
    for levels in a.levels:
        torch.manual_seed(0)
        ch = [int(c) for c in a.channels.split(",")][:levels]
        net = UNet(nside, levels, ch, norm="pixel", scale=float(m.std())).to(dev)
        torch.nn.init.normal_(net.head.weight, std=0.1)
        with torch.no_grad():
            r_full = (m[None] - net(m[None]))[0]
        torch.cuda.synchronize()
        t0 = time.time()
        pt = Patches(ids, nside, levels)
        torch.cuda.synchronize()
        t_geo = time.time() - t0
        flat = torch.cat([m[i] for i in ids])[None]
        with torch.no_grad():
            r_patch = (flat - net.forward_patches(flat, pt))[0]
        rms = float(r_full.square().mean().sqrt())
        edges = np.arange(0, a.radius + 1e-9, 1.0)
        err = {k: [] for k in range(len(edges) - 1)}
        for c, i, rp in zip(centres, ids, pt.split(r_patch)):
            dist = torch.rad2deg(torch.acos((vecs[i] @ vecs[c]).clamp(-1, 1))).cpu().numpy()
            e = (rp - r_full[i]).abs().cpu().numpy() / rms
            for k in range(len(edges) - 1):
                sel = (dist >= edges[k]) & (dist < edges[k + 1])
                if sel.any():
                    err[k].append(e[sel].max())
        prof = " ".join(f"{edges[k]:.0f}-{edges[k + 1]:.0f}:{max(v):.1e}" for k, v in err.items() if v)
        print(f"levels {levels}: max |r_patch - r_sky| / rms(r_sky) per ring [deg]: {prof}", flush=True)

        x = flat.clone().requires_grad_(True)
        for it in range(4):
            if it == 1:
                torch.cuda.synchronize()
                t0 = time.time()
            net.forward_patches(x, pt).square().sum().backward()
        torch.cuda.synchronize()
        t_patch = (time.time() - t0) / 3
        with torch.no_grad():
            for it in range(4):
                if it == 1:
                    torch.cuda.synchronize()
                    t0 = time.time()
                net(m[None])
            torch.cuda.synchronize()
            t_sky = (time.time() - t0) / 3
        print(f"levels {levels}, channels {ch}: geometry {t_geo:.2f} s; forward+backward of the patches "
              f"{t_patch * 1e3:.0f} ms; forward of the full sky {t_sky * 1e3:.0f} ms; "
              f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
