"""U-Net on the HEALPix NESTED grid: observed map d -> denoised map u.

The map is laid out as 12 base-face images (`healpix_faces`); the NESTED
parent mean is the 2 x 2 average pooling of a face image, so the encoder
goes down the HEALPix pyramid by pooling and the decoder comes back up by
copying every parent to its 4 children.  Every 3 x 3 convolution pads its
input with a one-pixel ring taken from the neighbouring faces (padding of
STL-Dev-MPI `diffusion/model.py`); channels are scalars (no re-orientation
across face edges).

The output is a correction of the input, `u = d - r(d)`, with the last
layer of `r` initialised to zero: an untrained network is the identity.
The network works on maps standardised by `scale` (the std of the noisy
map), in float32.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import healpix_faces as hf


class Padding:
    """One-pixel ring from the neighbouring faces, index tables per face size."""

    def __init__(self):
        self.tables = {}

    def __call__(self, x):
        b12, c, n, _ = x.shape
        key = (n, x.device)
        if key not in self.tables:
            self.tables[key] = torch.as_tensor(hf.pad_flat(n, 1).ravel(), device=x.device)
        idx = self.tables[key]
        b = b12 // 12
        y = x.reshape(b, 12, c, n * n).permute(0, 2, 1, 3).reshape(b, c, 12 * n * n)
        y = y[:, :, idx].reshape(b, c, 12, n + 2, n + 2)
        return y.permute(0, 2, 1, 3, 4).reshape(b12, c, n + 2, n + 2)


class Conv3(nn.Module):
    def __init__(self, pad, cin, cout):
        super().__init__()
        self.pad = pad
        self.conv = nn.Conv2d(cin, cout, 3)

    def forward(self, x):
        return self.conv(self.pad(x))


def block(pad, cin, cout, slope):
    return nn.Sequential(Conv3(pad, cin, cout), nn.GroupNorm(min(8, cout), cout), nn.LeakyReLU(slope),
                         Conv3(pad, cout, cout), nn.GroupNorm(min(8, cout), cout), nn.LeakyReLU(slope))


def default_channels(levels):
    return [32, 64, 96, 128, 128, 128, 128][:levels]


class UNet(nn.Module):
    """`levels` resolutions from `nside` down to `nside / 2^(levels-1)`."""

    def __init__(self, nside, levels=4, channels=None, slope=0.2):
        super().__init__()
        assert nside >> (levels - 1) >= 1, "too many levels for nside"
        self.nside, self.levels = nside, levels
        ch = list(channels or default_channels(levels))
        assert len(ch) == levels, "one channel count per level"
        self.channels = ch
        self.pad = Padding()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for k, c in enumerate(ch):
            self.enc.append(block(self.pad, 1 if k == 0 else ch[k - 1], c, slope))
        for k in range(levels - 2, -1, -1):
            self.dec.append(block(self.pad, ch[k + 1] + ch[k], ch[k], slope))
        self.head = nn.Conv2d(ch[0], 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        fn = hf.face_nest(nside).ravel()
        self.register_buffer("face_of_nest", torch.as_tensor(np.argsort(fn)), persistent=False)
        self.register_buffer("nest_of_face", torch.as_tensor(fn), persistent=False)

    def config(self):
        return {"nside": self.nside, "levels": self.levels, "channels": self.channels}

    def to_faces(self, m):
        """NESTED maps `[B, 12 n^2]` -> face images `[B 12, 1, n, n]`."""
        n = self.nside
        return m[:, self.nest_of_face].reshape(-1, 1, n, n)

    def to_nested(self, img):
        """Face images `[B 12, 1, n, n]` -> NESTED maps `[B, 12 n^2]`."""
        b = img.shape[0] // 12
        return img.reshape(b, 12 * self.nside**2)[:, self.face_of_nest]

    def forward(self, d, scale):
        """`d`: NESTED maps `[B, 12 nside^2]`; returns `u` with the dtype of `d`."""
        x = self.to_faces((d / scale).float())
        skips = []
        for k, enc in enumerate(self.enc):
            x = enc(x if k == 0 else F.avg_pool2d(x, 2))
            skips.append(x)
        for i, dec in enumerate(self.dec):
            k = self.levels - 2 - i
            x = dec(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), skips[k]], 1))
        r = self.to_nested(self.head(x)).to(d.dtype)
        return d - scale * r


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = UNet(ck["nside"], ck["levels"], ck["channels"]).to(device)
    net.load_state_dict(ck["state"])
    return net.eval(), ck
