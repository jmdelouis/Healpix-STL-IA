"""Disc patches of the HEALPix NESTED grid and their pyramid, for training the
U-Net on patches.

A patch is every pixel whose centre lies within an angular radius of a centre
drawn at random among the pixels: a true disc, not a set of NESTED cells, so
that no pixel shape or fixed cell boundary enters the training.  Its pyramid
keeps, at every coarser level, the parents with at least one child in the
patch, the parent value being the mean of the children present.  The STL
stencil of every level is bound to the pixels present: taps falling outside
the patch get weight zero and the 4 weights of each tap are renormalised over
the pixels present (a tap with none of its pixels present reads the pixel
itself), as the STL engine's `bind_support` does on partial grids.

Several patches are concatenated into one flat set of pixels (block-diagonal
gather matrices), so that a batch of patches is one call of the network.
"""

import math

import torch

from . import stl_torch_geometry as tg


def pixel_vectors(nside, device):
    """Unit vectors `[npix, 3]` of the NESTED pixel centres."""
    order = int(round(math.log2(nside)))
    ids = torch.arange(12 * nside * nside, device=device)
    ix, iy, face = tg.nest2xyf(ids, order)
    theta, phi = tg.xyf2ang(ix, iy, face, nside)
    theta, phi = theta.to(torch.float64), phi.to(torch.float64)
    return torch.stack([torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)], 1)


def disc_ids(vecs, centre, radius):
    """Sorted NESTED ids of the pixels within `radius` (radians) of pixel `centre`."""
    return torch.nonzero(vecs @ vecs[centre] >= math.cos(radius))[:, 0]


def stencil(ids, nside, ksz, gauge):
    """NESTED ids `[m]` -> (`[m, k^2, 4]` neighbour ids, `[m, k^2, 4]` bilinear weights)."""
    order = int(round(math.log2(nside)))
    ix, iy, face = tg.nest2xyf(ids, order)
    theta, phi = tg.xyf2ang(ix, iy, face, nside)
    rot = tg.rotation_matrices(theta.to(torch.float64), phi.to(torch.float64), 1, gauge, torch.float64)[:, 0]
    vec = tg.local_kernel_grid(ksz, nside, ids.device, torch.float64)
    taps = torch.einsum("mij,pj->mpi", rot, vec).reshape(-1, 3)
    nb, w = tg.interp_weights_nested(taps, order)
    p = ksz * ksz
    return nb.view(-1, p, 4).to(ids.device), w.view(-1, p, 4).to(ids.device, torch.float64)


_BOUND = {}
_BOUND_MAX = 4096


def bound_stencil(ids, nside, ksz, gauge):
    """Stencil of the sorted NESTED ids `[m]` bound to them: (`[m, k^2, 4]`
    positions in `ids`, renormalised weights).  Cached by disc (the training
    draws its centres from a finite set)."""
    key = (nside, ksz, gauge, ids.numel(), int(ids[0]), int(ids[-1]), int(ids.sum()), str(ids.device))
    if key in _BOUND:
        return _BOUND[key]
    device = ids.device
    nb, w = stencil(ids, nside, ksz, gauge)
    pos = torch.searchsorted(ids, nb.reshape(-1)).view_as(nb).clamp(max=len(ids) - 1)
    present = ids[pos] == nb
    w = w * present
    tot = w.sum(-1, keepdim=True)
    none = tot[..., 0] == 0
    w = torch.where(tot > 0, w / tot.clamp_min(1e-300), w)
    # a tap with none of its pixels present reads the pixel itself
    self_pos = torch.arange(len(ids), device=device)[:, None, None].expand_as(pos)
    pos = torch.where(none[..., None], self_pos, pos)
    w = torch.where(none[..., None], torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=w.dtype), w)
    pos, w = pos.to(torch.int32), w.to(torch.float32)  # compact: the cache holds thousands of discs
    if len(_BOUND) >= _BOUND_MAX:
        _BOUND.pop(next(iter(_BOUND)))
    _BOUND[key] = (pos, w)
    return pos, w


class Level:
    """One resolution of a batch of patches: the pixels (flat), the gather
    matrix of the stencil and its transpose, and the map to the parents."""

    def __init__(self, parts, nside, ksz, gauge):
        """`parts`: per patch, sorted NESTED ids at this nside."""
        device = parts[0].device
        self.p = ksz * ksz
        self.sizes = [len(i) for i in parts]
        self.n = sum(self.sizes)
        self.ids = parts
        rows, cols, vals = [], [], []
        off = 0
        for ids in parts:
            pos, w = bound_stencil(ids, nside, ksz, gauge)
            m = len(ids)
            rows.append((torch.arange(m * self.p, device=device)[:, None].expand(-1, 4) + off * self.p).reshape(-1))
            cols.append((pos.long() + off).reshape(-1))
            vals.append(w.reshape(-1).float())
            off += m
        coo = torch.sparse_coo_tensor(torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals),
                                      (self.n * self.p, self.n)).coalesce()
        self.S = coo.to_sparse_csr()
        self.St = coo.t().coalesce().to_sparse_csr()

    def parents(self):
        """Per patch, the sorted NESTED ids of the parents (one level coarser)."""
        return [torch.unique(ids // 4) for ids in self.ids]

    def link(self, coarse):
        """Index of the parent of every pixel in the flat pixels of `coarse`, and child counts."""
        idx, off = [], 0
        for ids, par in zip(self.ids, coarse.ids):
            idx.append(torch.searchsorted(par, ids // 4) + off)
            off += len(par)
        self.parent = torch.cat(idx)
        self.count = torch.zeros(coarse.n, device=self.parent.device).index_add_(
            0, self.parent, torch.ones(self.n, device=self.parent.device))


class Patches:
    """Pyramid of a batch of disc patches: `levels` resolutions from `nside`."""

    def __init__(self, ids, nside, levels, ksz=3, gauge="phi"):
        parts = list(ids)
        self.levels = []
        for k in range(levels):
            lv = Level(parts, nside >> k, ksz, gauge)
            self.levels.append(lv)
            parts = lv.parents()
        for fine, coarse in zip(self.levels[:-1], self.levels[1:]):
            fine.link(coarse)
        self.ids = list(ids)
        self.offsets = [0]
        for i in ids:
            self.offsets.append(self.offsets[-1] + len(i))

    def pool(self, x, k):
        """`[B, C, n_k]` -> `[B, C, n_{k+1}]`: mean of the children present."""
        lv = self.levels[k]
        out = torch.zeros(x.shape[0], x.shape[1], self.levels[k + 1].n, device=x.device, dtype=x.dtype)
        out.index_add_(2, lv.parent, x)
        return out / lv.count.to(x.dtype)

    def up(self, x, k):
        """`[B, C, n_{k+1}]` -> `[B, C, n_k]`: every pixel takes its parent value."""
        return x[:, :, self.levels[k].parent]

    def split(self, x):
        """Flat `[..., n_0]` -> list of per-patch tensors."""
        return [x[..., a:b] for a, b in zip(self.offsets[:-1], self.offsets[1:])]
