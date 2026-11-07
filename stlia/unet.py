"""U-Net on the HEALPix NESTED grid: observed map d -> denoised map u.

Convolutions (`conv="sphere"`, the default): the STL stencil, as in
STL-Dev and FOSCAT.  For every pixel, the `k x k` taps of the kernel are
laid in the tangent plane (spacing: the pixel size `sqrt(4 pi / npix)`),
rotated to the pixel and its local gauge, and each tap is the bilinear
interpolation of its 4 nearest pixels (`stl_torch_geometry`, the torch port
of the STL geometry).  The same operator applies everywhere on the sphere:
there are no base faces and no edges.  The gauge sets the orientation of
the kernel: 'phi' follows the local meridian everywhere (singular only at
the two poles), 'cosmo' is the gauge of the STL engine (continuous through
the poles, flipped between the hemispheres).

`conv="faces"`: the first prototype, 3 x 3 convolutions on the 12 base-face
images with a one-pixel ring from the neighbouring faces (scalar features
not re-oriented across edges, no corner at the 8 three-face vertices); it
leaves artefacts along the face edges and is kept for comparison.

The encoder goes down the NESTED pyramid by the mean of the 4 children, the
decoder comes back up by copying every parent to its children.  The output
is a correction of the input, `u = d - r(d)`, with the last layer of `r`
initialised to zero: an untrained network is the identity.  The network
works on maps standardised by `scale` (the std of the noisy map), in
float32.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import healpix_faces as hf
from . import stl_torch_geometry as tg


class SphereLevel:
    """Gather matrix `S` `[npix k^2, npix]` of the stencil at one nside, and its transpose."""

    def __init__(self, nside, ksz, gauge, device):
        order = int(round(math.log2(nside)))
        npix = 12 * nside * nside
        ids = torch.arange(npix, device=device)
        ix, iy, face = tg.nest2xyf(ids, order)
        theta, phi = tg.xyf2ang(ix, iy, face, nside)
        rot = tg.rotation_matrices(theta.to(torch.float64), phi.to(torch.float64), 1, gauge, torch.float64)[:, 0]
        vec = tg.local_kernel_grid(ksz, nside, device, torch.float64)
        taps = torch.einsum("mij,pj->mpi", rot, vec).reshape(-1, 3)
        nb, w = tg.interp_weights_nested(taps, order)
        p = ksz * ksz
        rows = torch.arange(npix * p, device=device)[:, None].expand(-1, 4).reshape(-1)
        coo = torch.sparse_coo_tensor(torch.stack([rows, nb.reshape(-1).to(device)]),
                                      w.reshape(-1).to(device, torch.float32), (npix * p, npix)).coalesce()
        self.S = coo.to_sparse_csr()
        self.St = coo.t().coalesce().to_sparse_csr()
        self.npix, self.p = npix, p


class _Gather(torch.autograd.Function):
    """`S @ x` with the transpose precomputed for the backward pass."""

    @staticmethod
    def forward(ctx, x, S, St):
        ctx.St = St
        return torch.sparse.mm(S, x)

    @staticmethod
    def backward(ctx, g):
        return torch.sparse.mm(ctx.St, g.contiguous()), None, None


class SphereConv(nn.Module):
    """`[B, cin, npix]` -> `[B, cout, npix]` with the STL stencil of `level`."""

    def __init__(self, cin, cout, ksz=3):
        super().__init__()
        p = ksz * ksz
        self.weight = nn.Parameter(torch.empty(cout, p * cin))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(cout))
        self.p, self.cin = p, cin

    def forward(self, x, level):
        b, c, n = x.shape
        flat = x.permute(2, 0, 1).reshape(n, b * c)
        taps = _Gather.apply(flat, level.S, level.St)  # [n p, b c]
        taps = taps.view(n, self.p, b, c).permute(2, 0, 1, 3).reshape(b, n, self.p * c)
        return F.linear(taps, self.weight, self.bias).transpose(1, 2)


class PixelNorm(nn.Module):
    """Normalisation over the channels of every pixel (`[B, C, N]`): the
    network stays local, so that it is the same on a patch and on the sky."""

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c, 1))
        self.bias = nn.Parameter(torch.zeros(c, 1))

    def forward(self, x):
        x = (x - x.mean(1, keepdim=True)) * torch.rsqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        return x * self.weight + self.bias


def make_norm(kind, c):
    """Normalisation layer: ``"group"`` (GroupNorm over the map) or ``"pixel"``
    (:class:`PixelNorm`, over the channels of every pixel)."""
    return PixelNorm(c) if kind == "pixel" else nn.GroupNorm(min(8, c), c)


class SphereBlock(nn.Module):
    """Two sphere convolutions, each followed by a normalisation and a leaky
    ReLU: the block of every level of the encoder and of the decoder."""

    def __init__(self, cin, cout, ksz, slope, norm="group"):
        super().__init__()
        self.c1, self.c2 = SphereConv(cin, cout, ksz), SphereConv(cout, cout, ksz)
        self.n1, self.n2 = make_norm(norm, cout), make_norm(norm, cout)
        self.act = nn.LeakyReLU(slope)

    def forward(self, x, level):
        x = self.act(self.n1(self.c1(x, level)))
        return self.act(self.n2(self.c2(x, level)))


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


def face_block(pad, cin, cout, slope):
    return nn.Sequential(Conv3(pad, cin, cout), nn.GroupNorm(min(8, cout), cout), nn.LeakyReLU(slope),
                         Conv3(pad, cout, cout), nn.GroupNorm(min(8, cout), cout), nn.LeakyReLU(slope))


def default_channels(levels):
    """Default channels per level: 32, 64, 96, 128, then 128."""
    return [32, 64, 96, 128, 128, 128, 128][:levels]


class UNet(nn.Module):
    """Spherical U-Net on the HEALPix NESTED grid, d -> u = d - scale * r(d / scale).

    Parameters
    ----------
    nside : int
        Resolution of the input maps.
    levels : int
        Resolutions of the pyramid, from `nside` down to `nside / 2^(levels-1)`.
    channels : list of int, optional
        Channels per level (default :func:`default_channels`).
    slope : float
        Negative slope of the leaky ReLU.
    conv : {"sphere", "faces"}
        STL stencil on the sphere, or planar convolutions on the base faces
        (first prototype, kept for comparison).
    gauge : {"phi", "cosmo"}
        Orientation of the kernels on the sphere.
    ksz : int
        Kernel size (`ksz x ksz` taps).
    norm : {"group", "pixel"}
        Normalisation; "pixel" is required to train on patches.
    scale : float, optional
        Fixed input scale; None uses the std of every map.
    cond : int
        Number of conditioning maps (e.g. log I_857, a tracer of the dust
        intensity, hence of the local signal-to-noise ratio).  They are
        concatenated to the input and, averaged down the pyramid, to the
        input of every decoder level, so that the decoding depends on them.
        Sphere convolutions only.
    maps : int
        Number of maps denoised together: 1 (a scalar map, `d` `[B, npix]`)
        or 2 (a (Q, U) pair, `d` `[B, 2, npix]`, both in the input and in
        the output, sharing one input scale).  Sphere convolutions only.

    Notes
    -----
    The last layer starts at zero: an untrained network is the identity.
    """

    def __init__(self, nside, levels=4, channels=None, slope=0.2, conv="sphere", gauge="phi", ksz=3, norm="group",
                 scale=None, cond=0, maps=1):
        """`norm`: 'group' (GroupNorm over the map) or 'pixel' (over the
        channels of every pixel: required to train on patches and apply on the
        sky).  `scale`: fixed input scale (uK); None uses the std of every map."""
        super().__init__()
        assert nside >> (levels - 1) >= 1, "too many levels for nside"
        assert conv in ("sphere", "faces")
        assert conv == "sphere" or norm == "group", "face images: GroupNorm only"
        self.nside, self.levels, self.conv, self.gauge, self.ksz = nside, levels, conv, gauge, ksz
        self.norm, self.scale, self.cond, self.maps = norm, scale, int(cond), int(maps)
        assert self.cond == 0 or conv == "sphere", "conditioning maps: sphere convolutions only"
        assert self.maps == 1 or conv == "sphere", "joint maps: sphere convolutions only"
        ch = list(channels or default_channels(levels))
        assert len(ch) == levels, "one channel count per level"
        self.channels = ch
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        if conv == "sphere":
            for k, c in enumerate(ch):
                self.enc.append(SphereBlock(self.maps + self.cond if k == 0 else ch[k - 1], c, ksz, slope, norm))
            for k in range(levels - 2, -1, -1):
                self.dec.append(SphereBlock(ch[k + 1] + ch[k] + self.cond, ch[k], ksz, slope, norm))
            self.head = nn.Conv1d(ch[0], self.maps, 1)
            self.geometry = None
        else:
            self.pad = Padding()
            for k, c in enumerate(ch):
                self.enc.append(face_block(self.pad, 1 if k == 0 else ch[k - 1], c, slope))
            for k in range(levels - 2, -1, -1):
                self.dec.append(face_block(self.pad, ch[k + 1] + ch[k], ch[k], slope))
            self.head = nn.Conv2d(ch[0], 1, 1)
            fn = hf.face_nest(nside).ravel()
            self.register_buffer("face_of_nest", torch.as_tensor(np.argsort(fn)), persistent=False)
            self.register_buffer("nest_of_face", torch.as_tensor(fn), persistent=False)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def config(self):
        """Constructor arguments, saved with the weights in ``model.pt``."""
        return {"nside": self.nside, "levels": self.levels, "channels": self.channels, "conv": self.conv,
                "gauge": self.gauge, "ksz": self.ksz, "norm": self.norm, "scale": self.scale,
                "cond": self.cond, "maps": self.maps}

    def input_scale(self, d):
        """Scale of the input: the fixed `scale`, or the std of `d`."""
        return d.std() if self.scale is None else torch.as_tensor(self.scale, dtype=d.dtype, device=d.device)

    def forward_patches(self, d, patches, cond=None):
        """`d`: `[B, n]` values on the flat pixels of `patches`
        (`stlia.regions.Patches`, levels as the network; `[B, 2, n]` with
        `maps` = 2), `cond` `[B, cond, n]` the conditioning maps on the same
        pixels; returns `u`, shaped like `d`."""
        assert self.conv == "sphere" and self.norm == "pixel", "patches need sphere convolutions and pixel norm"
        assert d.dim() == (2 if self.maps == 1 else 3), "patches: d is [B, n], or [B, maps, n]"
        scale = self.input_scale(d)
        x = (d / scale).float()
        if self.maps == 1:
            x = x[:, None]
        conds = self._cond_levels(cond, lambda c, k: patches.pool(c, k))
        if conds:
            x = torch.cat([x, conds[0]], 1)
        skips = []
        for k, enc in enumerate(self.enc):
            if k:
                x = patches.pool(x, k - 1)
            x = enc(x, patches.levels[k])
            skips.append(x)
        for i, dec in enumerate(self.dec):
            k = self.levels - 2 - i
            parts = [patches.up(x, k), skips[k]] + ([conds[k]] if conds else [])
            x = dec(torch.cat(parts, 1), patches.levels[k])
        r = self.head(x).to(d.dtype)
        return d - scale * (r[:, 0] if self.maps == 1 else r)

    def stencils(self, device):
        """Stencils of every level, built on first use on the device of the maps."""
        if self.geometry is None or self.geometry[0].S.device != device:
            self.geometry = [SphereLevel(self.nside >> k, self.ksz, self.gauge, device) for k in range(self.levels)]
        return self.geometry

    def _cond_levels(self, cond, pool):
        """The conditioning maps `[B, cond, n]` at every level (float32), or []."""
        if self.cond == 0:
            return []
        assert cond is not None and cond.shape[1] == self.cond, f"the network needs {self.cond} conditioning maps"
        out = [cond.float()]
        for k in range(self.levels - 1):
            out.append(pool(out[-1], k))
        return out

    def forward(self, d, scale=None, cond=None):
        """`d`: NESTED maps `[B, 12 nside^2]` (`[B, 2, 12 nside^2]` with
        `maps` = 2); returns `u`, shaped and typed like `d`.
        `scale` overrides the input scale of the network; `cond`
        `[B, cond, 12 nside^2]`: the conditioning maps (see the class)."""
        if scale is None:
            scale = self.input_scale(d)
        x = (d / scale).float()
        if self.conv == "sphere":
            geo = self.stencils(x.device)
            if self.maps == 1:
                x = x[:, None]
            conds = self._cond_levels(cond, lambda c, k: c.view(c.shape[0], c.shape[1], -1, 4).mean(-1))
            if conds:
                x = torch.cat([x, conds[0]], 1)
            skips = []
            for k, enc in enumerate(self.enc):
                if k:
                    x = x.view(x.shape[0], x.shape[1], -1, 4).mean(-1)
                x = enc(x, geo[k])
                skips.append(x)
            for i, dec in enumerate(self.dec):
                k = self.levels - 2 - i
                parts = [x.repeat_interleave(4, dim=-1), skips[k]] + ([conds[k]] if conds else [])
                x = dec(torch.cat(parts, 1), geo[k])
            r = self.head(x)
            if self.maps == 1:
                r = r[:, 0]
        else:
            n = self.nside
            x = x[:, self.nest_of_face].reshape(-1, 1, n, n)
            skips = []
            for k, enc in enumerate(self.enc):
                x = enc(x if k == 0 else F.avg_pool2d(x, 2))
                skips.append(x)
            for i, dec in enumerate(self.dec):
                k = self.levels - 2 - i
                x = dec(torch.cat([F.interpolate(x, scale_factor=2, mode="nearest"), skips[k]], 1))
            r = self.head(x).reshape(-1, 12 * n * n)[:, self.face_of_nest]
        return d - scale * r.to(d.dtype)


def load(path, device):
    """Load a network saved by the training commands.

    Returns
    -------
    net : UNet
        The network, in evaluation mode, on `device`.
    checkpoint : dict
        The saved dictionary: weights, :meth:`UNet.config`, and the run
        metadata (``j``, ``noise_rms``)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    net = UNet(ck["nside"], ck["levels"], ck["channels"], conv=ck.get("conv", "faces"),
               gauge=ck.get("gauge", "phi"), ksz=ck.get("ksz", 3), norm=ck.get("norm", "group"),
               scale=ck.get("scale"), cond=ck.get("cond", 0), maps=ck.get("maps", 1)).to(device)
    net.load_state_dict(ck["state"])
    return net.eval(), ck
