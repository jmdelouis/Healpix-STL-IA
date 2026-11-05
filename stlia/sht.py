"""Differentiable spherical harmonic transforms and power spectra of full-sky
HEALPix maps (torch), for spectral losses.

Adapted from ``healpix_analyse.healpix_sht`` (GRID4EARTH/healpix-analyse,
Apache-2.0), itself adapted from the ring-FFT transform of FOSCAT
(``foscat.alm``).  Changes: the ring geometry and the ring <-> nested
permutations come from healpy instead of healpix-geo (sphere only), the
synthesis tables are built on first use, and ``HEALPixSHT`` takes `nside`.

Ring-based Spherical Harmonic Transform (SHT) for full-sky HEALPix maps.

Algorithm
---------
Adapts the FOSCAT ring-FFT algorithm to the healpix-analyse package.
By exploiting the HEALPix ring structure, the number of Legendre summations
is reduced from 12·nside² (all-pixel, alm_latlon.py) to 4·nside-1 (one per
iso-latitude ring), giving a ~3·nside speed-up at fixed lmax.

For each iso-latitude ring r with N_r uniformly-spaced pixels at colatitude θ_r
and starting longitude φ_0^r:

  (1)  F_r(m) = FFT_m(f_ring_r) × exp(-i·m·φ_0^r)      [phase-corrected ring FFT]
  (2)  a_lm   = sqrt(2l+1)/N × Σ_r  √4π·P̃_lm(cos θ_r) × F_r(m)

where N = 12·nside².  This follows the standard orthonormal convention
(same as healpy and alm_latlon.py):

    Y_lm(θ,φ) = sqrt((2l+1)/(4π)) · P̃_lm(cos θ) · exp(i·m·φ)
    a_lm       = ∫ f(θ,φ) · Y_lm*(θ,φ) dΩ

Differentiability
-----------------
All hot-path operations use torch.fft.fft/ifft and torch.einsum, which are
supported by torch.autograd.  Geometry (Legendre tables, phase matrices,
permutation indices) is precomputed once and stored as non-grad constants.

Spin support  (requires quaternionic + spherical)
-------------------------------------------------
spin=0 : scalar (temperature) field
spin=1 : vector field (curl/divergence decomposition)
spin=2 : CMB polarisation Q/U → E/B modes

The relation between (Q, U) and the spin-weighted harmonics sY_lm is:

    (Q ± iU)(n) = Σ_lm  [±almE ∓ i·almB] · ±sY_lm(n)

which gives:

    almE = -(Σ+ + Σ-) / 2
    almB =  (Σ+ - Σ-) / (2i)

where  Σ+(m) = Σ_r +sY_lm*(θ_r)·F_r^+(m),  F_r^+ = FFT(Q + iU),
and    Σ-(m) = Σ_r -sY_lm*(θ_r)·F_r^-(m),  F_r^- = FFT(Q - iU).

Public API
----------
HEALPixSHT(nside, lmax, dtype, device)

.map2alm(im, nest=False)                   → Tensor [..., K]
.alm2map(alm, nest=False)                  → Tensor [..., N]
.map2alm_spin(Q, U, spin, nest)            → (almE, almB)  each [..., K]
.alm2map_spin(almE, almB, spin, nest)      → (Q, U)        each [..., N]
.anafast(im, map2, spin, nest)             → Tensor [..., lmax+1]

Flat alm layout (same as healpy / alm_latlon.py):
    ``[m=0: l=0..lmax | m=1: l=1..lmax | … | m=lmax: l=lmax]``
    K = (lmax+1)·(lmax+2)//2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import healpy as hp

try:
    import quaternionic
    import spherical as _spherical_pkg
    _HAS_SPHERICAL = True
except ImportError:
    _HAS_SPHERICAL = False

# How the spin-weighted harmonic tables are built: "recursion" (Wigner-d recurrence in
# numpy, the default) or "spherical" (the packages quaternionic and spherical, much slower).
_SPIN_TABLES = "recursion"


ArrayLike = Union[np.ndarray, torch.Tensor]


# ===========================================================================
# Normalised associated Legendre polynomials
# ===========================================================================

def _double_factorial_log(n: int) -> float:
    """log(n!!) with convention 0!! = 1."""
    if n <= 0:
        return 0.0
    result, i = 0.0, n
    while i > 0:
        result += math.log(i)
        i -= 2
    return result


def _compute_legendre_m(
    x: np.ndarray,
    m: int,
    lmax: int,
    limit_range: float = 1e10,
) -> np.ndarray:
    """
    Compute √4π · P̃_lm(x) for l = m…lmax at all ring colatitudes at once.

    Same as _compute_legendre_m in alm_latlon.py — kept identical so both
    modules stay numerically consistent.

    Parameters
    ----------
    x          : (R,)         cos θ for each ring
    m          : azimuthal order
    lmax       : maximum degree
    limit_range: overflow guard for the three-term recurrence

    Returns
    -------
    result : (lmax-m+1, R) complex128
    """
    n_l   = lmax - m + 1
    R     = x.shape[0]
    res   = np.zeros((n_l, R), dtype=np.float64)
    ratio = np.zeros((n_l, 1), dtype=np.float64)
    log_lim = math.log(limit_range)
    inv_lim = 1.0 / limit_range

    # seed log-ratio for P_{mm}
    ratio[0, 0] = (
        _double_factorial_log(2 * m - 1)
        - 0.5 * float(np.sum(np.log(1.0 + np.arange(2 * m))))
        if m > 0
        else _double_factorial_log(2 * m - 1)
    )

    # P_{mm}(x)
    Pmm = (
        np.ones(R, dtype=np.float64)
        if m == 0
        else (0.5 - m % 2) * 2.0 * (1.0 - x ** 2) ** (m / 2.0)
    )
    res[0] = Pmm

    if m == lmax:
        return res * np.exp(ratio) * math.sqrt(4.0 * math.pi) + 0j

    # P_{m+1,m}(x)
    res[1]      = x * (2 * m + 1) * res[0]
    ratio[1, 0] = ratio[0, 0] - 0.5 * math.log(2 * m + 1)

    # three-term recurrence for l = m+2 … lmax
    for ell in range(m + 2, lmax + 1):
        res[ell - m] = (
            (2 * ell - 1) * x * res[ell - m - 1]
            - (ell + m - 1) * res[ell - m - 2]
        ) / (ell - m)
        ratio[ell - m, 0] = (
            0.5 * math.log(ell - m)
            - 0.5 * math.log(ell + m)
            + ratio[ell - m - 1, 0]
        )
        if np.max(np.abs(res[ell - m])) > limit_range:
            res[ell - m - 1] *= inv_lim
            res[ell - m]     *= inv_lim
            ratio[ell - m - 1, 0] += log_lim
            ratio[ell - m, 0]     += log_lim

    return res * np.exp(ratio) * math.sqrt(4.0 * math.pi) + 0j


def _wigner_d_rows(theta: np.ndarray, a: int, b: int, lmax: int, lmin: int) -> np.ndarray:
    """
    √((2l+1)/4π) · d^l_{a,b}(θ) for l = lmin…lmax (rows) at the colatitudes θ
    (columns), zero for l < max(|a|, |b|).

    d^l_{a,b} with the convention of the Wikipedia "Wigner D-matrix" formula
    (first index a, second b).  The start l0 = max(|a|, |b|) is the closed form,
    in logarithms; then the three-term recurrence in l, on values rescaled
    per ring so that neither the start (as small as sin(θ/2)^(2 l0)) nor the
    growth overflows.
    """
    R = theta.shape[0]
    out = np.zeros((lmax - lmin + 1, R), dtype=np.float64)
    l0 = max(abs(a), abs(b))
    assert l0 >= 1, "spin-weighted tables only (the recurrence divides by l)"
    if l0 > lmax:
        return out
    lc, ls = np.log(np.cos(theta / 2)), np.log(np.sin(theta / 2))
    lg = lambda n: math.lgamma(n + 1)  # log n!
    # d^{l0}_{a,b}: sum over k of (-1)^(a-b+k) sqrt((j+a)!(j-a)!(j+b)!(j-b)!) /
    # ((j+b-k)! k! (a-b+k)! (j-a-k)!) cos(θ/2)^(2j+b-a-2k) sin(θ/2)^(a-b+2k)
    j = l0
    pref = 0.5 * (lg(j + a) + lg(j - a) + lg(j + b) + lg(j - b))
    logs, signs = [], []
    for k in range(max(0, b - a), min(j + b, j - a) + 1):
        logs.append(pref - lg(j + b - k) - lg(k) - lg(a - b + k) - lg(j - a - k)
                    + (2 * j + b - a - 2 * k) * lc + (a - b + 2 * k) * ls)
        signs.append(-1.0 if (a - b + k) % 2 else 1.0)
    logs = np.array(logs)
    top = logs.max(0)
    val = (np.array(signs)[:, None] * np.exp(logs - top)).sum(0)  # d^{l0} = val · exp(top)
    scale = top.copy()
    prev = np.zeros(R)
    cur = val
    x = np.cos(theta)
    big, lbig = 1e150, math.log(1e150)
    for ell in range(l0, lmax + 1):
        if ell >= lmin:
            out[ell - lmin] = math.sqrt((2 * ell + 1) / (4 * math.pi)) * cur * np.exp(scale)
        if ell == lmax:
            break
        # d^{l+1} from d^l and d^{l-1}
        c1 = (2 * ell + 1) * (ell * (ell + 1) * x - a * b)
        c0 = (ell + 1) * math.sqrt(max(ell * ell - a * a, 0) * max(ell * ell - b * b, 0))
        den = ell * math.sqrt(((ell + 1) ** 2 - a * a) * ((ell + 1) ** 2 - b * b))
        prev, cur = cur, (c1 * cur - c0 * prev) / den
        over = np.abs(cur) > big
        if over.any():
            cur[over] /= big
            prev[over] /= big
            scale[over] += lbig
    return out


# ===========================================================================
# HEALPixSHT
# ===========================================================================

class HEALPixSHT:
    """
    Ring-based Spherical Harmonic Transform for full-sky HEALPix maps.

    Parameters
    ----------
    nside     : int            HEALPix resolution (a power of 2).
    lmax      : int or None    Maximum multipole.  Default: 3·nside - 1.
    dtype     : torch.dtype    torch.float32 or torch.float64.
    device    : str or device  Defaults to CUDA when available.
    """

    def __init__(
        self,
        nside: int,
        lmax:      Optional[int]               = None,
        dtype:     torch.dtype                 = torch.float32,
        device:    Optional[Union[str, torch.device]] = None,
    ) -> None:
        level = int(round(math.log2(nside)))
        if 2 ** level != nside:
            raise ValueError("nside must be a power of 2.")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64.")

        self.level     = int(level)
        self.nside     = 2 ** self.level
        self.lmax      = int(lmax) if lmax is not None else 3 * self.nside - 1
        self.dtype     = dtype
        self.cdtype    = (
            torch.complex64 if dtype == torch.float32 else torch.complex128
        )
        self.device    = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._depth  = self.level
        self.n_pix   = 12 * self.nside ** 2
        self.n_rings = 4 * self.nside - 1
        self.n_alm   = (self.lmax + 1) * (self.lmax + 2) // 2

        # precompute ring geometry
        (
            self._ring_theta,    # (R,) float64  colatitude per ring
            self._ring_phi0,     # (R,) float64  starting longitude per ring
            self._ring_n,        # (R,) int64     pixels per ring
            self._ring_start,    # (R,) int64     first pixel index per ring
            self._ring2nest_np,  # (N,) int64     ring→nested permutation
            self._nest2ring_np,  # (N,) int64     nested→ring permutation
        ) = self._build_ring_geometry()

        # phase shift matrix: exp(-i·m·φ_0^r),  shape (R, lmax+1)
        self._phase = self._build_phase_matrix()

        # Legendre tables for spin-0
        # _plm_ana[m]: (lmax-m+1, R)  —  sqrt(2l+1)/N × √4π·P̃_lm  (analysis)
        # _plm_syn[m]: (lmax-m+1, R)  —  sqrt(2l+1)/(4π) × √4π·P̃_lm (synthesis)
        self._plm_ana: List[torch.Tensor] = []
        self._plm_syn: List[torch.Tensor] = []
        self._build_legendre_tables()

        # spin-weighted harmonic tables (lazily filled by _ensure_spin)
        # _spin_Yp[s][m], _spin_Ym[s][m]       — with pixel weight (analysis)
        # _spin_Yp_syn[s][m], _spin_Ym_syn[s][m] — without (synthesis)
        self._spin_Yp:     Dict[int, Dict[int, torch.Tensor]] = {}
        self._spin_Ym:     Dict[int, Dict[int, torch.Tensor]] = {}
        self._spin_Yp_syn: Dict[int, Dict[int, torch.Tensor]] = {}
        self._spin_Ym_syn: Dict[int, Dict[int, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Precomputation helpers
    # ------------------------------------------------------------------

    def _build_ring_geometry(self) -> Tuple:
        """Compute per-ring arrays and ring ↔ nested permutations."""
        nside  = self.nside
        depth  = self._depth
        R      = self.n_rings
        N      = self.n_pix

        ring_start = np.empty(R, dtype=np.int64)
        ring_n     = np.empty(R, dtype=np.int64)
        n = 0
        # north polar cap  (rings 0 … nside-2)
        for k in range(nside - 1):
            Nr = 4 * (k + 1)
            ring_start[k] = n;  ring_n[k] = Nr;  n += Nr
        # equatorial belt  (rings nside-1 … 3*nside-1)
        for k in range(2 * nside + 1):
            Nr = 4 * nside
            ring_start[nside - 1 + k] = n;  ring_n[nside - 1 + k] = Nr;  n += Nr
        # south polar cap  (rings 3*nside … 4*nside-2)
        for k in range(nside - 1):
            Nr = 4 * (nside - 1 - k)
            ring_start[3 * nside + k] = n;  ring_n[3 * nside + k] = Nr;  n += Nr

        # (θ_r, φ_0^r) from the first pixel of each ring (RING ordering)
        ring_theta, ring_phi0 = hp.pix2ang(nside, ring_start)
        ring_theta = np.asarray(ring_theta, dtype=np.float64)
        ring_phi0 = np.asarray(ring_phi0, dtype=np.float64)

        all_ring = np.arange(N, dtype=np.int64)
        ring2nest = np.asarray(hp.ring2nest(nside, all_ring), dtype=np.int64)
        nest2ring = np.empty_like(ring2nest)
        nest2ring[ring2nest] = all_ring

        return ring_theta, ring_phi0, ring_n, ring_start, ring2nest, nest2ring

    def _build_phase_matrix(self) -> torch.Tensor:
        """
        exp(-i·m·φ_0^r)  for m = 0…lmax, r = 0…R-1.
        Shape: (R, lmax+1), complex.
        """
        m    = np.arange(self.lmax + 1, dtype=np.float64)   # (M,)
        phi0 = self._ring_phi0[:, None]                      # (R, 1)
        ph   = np.exp(-1j * phi0 * m[None, :])               # (R, M)
        return torch.tensor(ph, dtype=self.cdtype, device=self.device)

    def _build_legendre_tables(self, synthesis: bool = False) -> None:
        """
        Precompute spin-0 Legendre tables for all m = 0…lmax.

        _plm_ana[m][l-m, r] = sqrt(2l+1)/N  ×  √4π·P̃_lm(cos θ_r)
        _plm_syn[m][l-m, r] = sqrt(2l+1)/(4π) × √4π·P̃_lm(cos θ_r)
                             = sqrt((2l+1)/4π) · P̃_lm(cos θ_r)   ← = Y_lm(θ,0)

        The analysis tables are built with the transform; the synthesis
        tables (``synthesis=True``) only when ``alm2map`` is first called,
        since a power spectrum does not need them (each set holds about
        lmax²·R/2 complex numbers: 0.5 GB in float32 at nside 256, lmax 511).
        """
        co_th = np.cos(self._ring_theta)   # (R,) float64
        lmax  = self.lmax
        N     = float(self.n_pix)
        target = self._plm_syn if synthesis else self._plm_ana

        for m in range(lmax + 1):
            plm_raw = _compute_legendre_m(co_th, m, lmax)   # (L, R) complex128
            l_vals  = np.arange(m, lmax + 1, dtype=np.float64)  # (L,)
            sqrt2lp1 = np.sqrt(2.0 * l_vals + 1.0)[:, None]     # (L, 1)
            norm = (4.0 * math.pi) if synthesis else N
            target.append(torch.tensor(plm_raw * (sqrt2lp1 / norm), dtype=self.cdtype, device=self.device))

    def _ensure_spin(self, spin: int) -> None:
        """
        Lazily precompute spin-s weighted spherical harmonics for |spin| > 0.

        Evaluated at (θ_r, φ=0) for each ring r using the quaternionic /
        spherical packages.  At φ=0 the spin harmonics are real, so we take
        the .real part (same convention as FOSCAT alm.py init_Ys).

        Analysis tables  (with pixel weight w = 4π/N):
            _spin_Yp[s][m]: (L, R)  — +s harmonics
            _spin_Ym[s][m]: (L, R)  — -s harmonics

        Synthesis tables (without pixel weight):
            _spin_Yp_syn[s][m]: (L, R)
            _spin_Ym_syn[s][m]: (L, R)
        """
        if spin in self._spin_Yp:
            return
        if _SPIN_TABLES == "recursion":
            self._ensure_spin_recursion(spin)
            return
        if not _HAS_SPHERICAL:
            raise ImportError(
                "Spin ≠ 0 transforms require 'quaternionic' and 'spherical'.\n"
                "Install with:  pip install quaternionic spherical"
            )

        lmax = self.lmax
        lth  = self._ring_theta                        # (R,) float64
        R    = len(lth)
        w    = 4.0 * math.pi / float(self.n_pix)      # equal-area pixel weight

        wigner = _spherical_pkg.Wigner(lmax)
        R_q    = quaternionic.array.from_spherical_coordinates(lth, np.zeros(R))

        # sYlm(s, R) returns shape (R_pts, K_alm) — transpose → (K_alm, R_pts).
        #
        # Convention (spherical / Boyle package):
        #   wigner.sYlm(s, R)[Yindex(l, m)] = sqrt((2l+1)/4π) · D^l_{m,-s}(R)*
        #
        # At φ=0 (R = R_y(θ)):  D^l_{m,-s}(R_y(θ))* = d^l_{m,-s}(θ)   (real)
        # At φ=0 for -s:        D^l_{m,s}(R_y(θ))*  = d^l_{m,s}(θ)    (real)
        #
        # Identified values at φ=0:
        #   wigner.sYlm(+s, R_y(θ))[Yindex(l,m)] = sqrt((2l+1)/4π) d^l_{m,-s}(θ)
        #                                         = +sY_lm*(θ, 0)   ← needed for Σ+
        #   wigner.sYlm(-s, R_y(θ))[Yindex(l,m)] = sqrt((2l+1)/4π) d^l_{m,s}(θ)
        #                                         = -sY_lm*(θ, 0)   ← needed for Σ-
        #
        # Therefore NO sign correction is needed.
        #
        # We take .real to discard the O(machine-ε) imaginary residuals that
        # accumulate in the Wigner D-matrix evaluation at exactly φ=0.
        # (Without .real, those residuals corrupt the einsum for spin=2 as well.)
        #
        # DO NOT multiply by (-1)^spin.  That correction negates BOTH +s and -s
        # tables simultaneously, which:
        #   - is a no-op for spin=2  ((-1)^2=1)  → misleading coincidence
        #   - negates almE and flips Im(almE) for spin=1, making it wrong
        # synthesis tables; the analysis ones are the same times the pixel weight w
        iplus  = np.ascontiguousarray(wigner.sYlm( spin, R_q).T.real)  # (K_alm, R) +s
        imoins = np.ascontiguousarray(wigner.sYlm(-spin, R_q).T.real)  # (K_alm, R) -s

        Yp_dict  = {};  Ym_dict  = {}
        Yps_dict = {};  Yms_dict = {}

        for m in range(lmax + 1):
            idx = np.array([wigner.Yindex(l, m) for l in range(m, lmax + 1)])

            Yps_dict[m] = torch.tensor(iplus[idx], device=self.device).to(self.cdtype)
            Yms_dict[m] = torch.tensor(imoins[idx], device=self.device).to(self.cdtype)
            Yp_dict[m]  = Yps_dict[m] * w
            Ym_dict[m]  = Yms_dict[m] * w
        del iplus, imoins

    def _ensure_spin_recursion(self, spin: int) -> None:
        """The tables of :meth:`_ensure_spin` from the Wigner-d recurrence
        (:func:`_wigner_d_rows`): +s table sqrt((2l+1)/4pi) d^l_{m,-s}(theta),
        -s table sqrt((2l+1)/4pi) d^l_{m,s}(theta), as the identification
        documented in :meth:`_ensure_spin`."""
        lmax = self.lmax
        lth = self._ring_theta
        w = 4.0 * math.pi / float(self.n_pix)
        Yp_dict, Ym_dict, Yps_dict, Yms_dict = {}, {}, {}, {}
        for m in range(lmax + 1):
            for sgn, syn, ana in ((-1, Yps_dict, Yp_dict), (1, Yms_dict, Ym_dict)):
                t = torch.tensor(_wigner_d_rows(lth, m, sgn * spin, lmax, m), device=self.device).to(self.cdtype)
                syn[m], ana[m] = t, t * w
        self._spin_Yp[spin] = Yp_dict
        self._spin_Ym[spin] = Ym_dict
        self._spin_Yp_syn[spin] = Yps_dict
        self._spin_Ym_syn[spin] = Yms_dict

        self._spin_Yp[spin]     = Yp_dict
        self._spin_Ym[spin]     = Ym_dict
        self._spin_Yp_syn[spin] = Yps_dict
        self._spin_Ym_syn[spin] = Yms_dict

    # ------------------------------------------------------------------
    # Ring FFT helpers  (differentiable)
    # ------------------------------------------------------------------

    def _ring_groups(self):
        """Rings grouped by length, cached: a list of (ring indices `[G]`, pixel
        indices `[G, Nr]`) on the device, for the batched per-ring FFTs."""
        if getattr(self, "_groups", None) is None:
            groups = {}
            for r in range(self.n_rings):
                groups.setdefault(int(self._ring_n[r]), []).append(r)
            self._groups = []
            for nr, rings in groups.items():
                pix = np.array([np.arange(self._ring_start[r], self._ring_start[r] + nr) for r in rings])
                self._groups.append((torch.as_tensor(rings, device=self.device),
                                     torch.as_tensor(pix, device=self.device)))
        return self._groups

    def _rings_to_fft(self, im: torch.Tensor) -> torch.Tensor:
        """
        Per-ring FFT with longitude phase correction.

        Parameters
        ----------
        im : (B, N) complex or real Tensor in RING order.

        Returns
        -------
        ft : (B, R, lmax+1) complex Tensor.
             ft[b, r, m] = FFT_m(ring_r) × exp(-i·m·φ_0^r)
        """
        M = self.lmax + 1
        B = im.shape[0]
        im = im if im.is_complex() else im.to(self.cdtype)
        ft = torch.zeros(B, self.n_rings, M, dtype=self.cdtype, device=im.device)
        modes = torch.arange(M, device=im.device)
        for rings, pix in self._ring_groups():
            F = torch.fft.fft(im[:, pix], dim=-1)           # (B, G, Nr)
            # short rings: mode m is the DFT bin m mod Nr (same as FOSCAT comp_tf)
            ft = ft.index_copy(1, rings, F[..., modes % pix.shape[1]])
        # Apply per-ring phase shift  exp(-i·m·φ_0^r)
        return ft * self._phase[None]                       # (B, R, lmax+1)

    def _synthesis_index(self):
        """Flat bins of the Hermitian ring spectra, cached: for every (r, m) the
        pixel index of the positive frequency m mod Nr, and for m >= 1 that of the
        negative frequency (Nr - m) mod Nr; and Nr per ring."""
        if getattr(self, "_syn_idx", None) is None:
            M = self.lmax + 1
            nr = self._ring_n.astype(np.int64)[:, None]
            st = self._ring_start.astype(np.int64)[:, None]
            m = np.arange(M)[None]
            pos = (st + m % nr).reshape(-1)
            neg = (st + (nr - m[:, 1:]) % nr).reshape(-1)
            self._syn_idx = (torch.as_tensor(pos, device=self.device), torch.as_tensor(neg, device=self.device),
                             torch.as_tensor(self._ring_n.astype(np.float64), device=self.device))
        return self._syn_idx

    def _fft_to_rings(self, ft_syn: torch.Tensor) -> torch.Tensor:
        """
        Per-ring IFFT: reconstruct pixel values from Fourier-synthesis coefficients.

        Adjoint of _rings_to_fft for real output maps.

        Parameters
        ----------
        ft_syn : (B, R, lmax+1) complex Tensor.
                 ft_syn[b, r, m] = Σ_l  a_lm · Ỹ_lm(θ_r)

        Returns
        -------
        im : (B, N) real Tensor in RING order.

        For a real map, f_j = G_0 + 2·Re[Σ_{m≥1} G_m e^{i2πmj/Nr}], G_m = ft_up[r, m]
        (a_{l,-m} = (-1)^m ā_lm).  The full Hermitian spectrum of every ring is
        H[m mod Nr] += Nr·G_m (all m) and H[(Nr - m) mod Nr] += Nr·Ḡ_m (m ≥ 1);
        aliased modes of short rings accumulate, which is correct.  Then
        IFFT(H) = f, ring by ring (batched over the rings of equal length).
        """
        B = ft_syn.shape[0]
        pos, neg, nr = self._synthesis_index()
        ft_up = ft_syn * self._phase[None].conj() * nr[None, :, None].to(ft_syn.real.dtype)  # Nr·G
        H = torch.zeros(B, self.n_pix, dtype=self.cdtype, device=ft_syn.device)
        H = H.index_add(1, pos, ft_up.reshape(B, -1))
        H = H.index_add(1, neg, ft_up[:, :, 1:].conj().reshape(B, -1))
        out = torch.zeros(B, self.n_pix, dtype=self.dtype, device=ft_syn.device)
        for _, pix in self._ring_groups():
            out = out.index_copy(1, pix.reshape(-1), torch.fft.ifft(H[:, pix], dim=-1).real.reshape(B, -1))
        return out

    # ------------------------------------------------------------------
    # alm flat-index helper
    # ------------------------------------------------------------------

    def _alm_slice(self, m: int) -> slice:
        """Slice for azimuthal order m in the flat alm vector."""
        lmax  = self.lmax
        start = m * (2 * lmax - m + 3) // 2
        return slice(start, start + lmax - m + 1)

    # ------------------------------------------------------------------
    # Permutation tensors (cached on demand)
    # ------------------------------------------------------------------

    def _r2n(self) -> torch.Tensor:
        """Ring→nested index tensor (long, on device)."""
        if not hasattr(self, "_r2n_t"):
            self._r2n_t = torch.as_tensor(
                self._ring2nest_np, dtype=torch.long, device=self.device
            )
        return self._r2n_t

    def _n2r(self) -> torch.Tensor:
        """Nested→ring index tensor (long, on device)."""
        if not hasattr(self, "_n2r_t"):
            self._n2r_t = torch.as_tensor(
                self._nest2ring_np, dtype=torch.long, device=self.device
            )
        return self._n2r_t

    # ------------------------------------------------------------------
    # Input coercions and validation
    # ------------------------------------------------------------------

    def _to_real(self, x: ArrayLike) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=self.dtype, device=self.device)
        return x.to(device=self.device, dtype=self.dtype)

    def _to_complex(self, x: ArrayLike) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=self.cdtype, device=self.device)
        return x.to(device=self.device, dtype=self.cdtype)

    def _check_map(self, x: torch.Tensor, name: str = "im") -> None:
        """Raise a clear error if the last dimension doesn't match n_pix."""
        if x.shape[-1] != self.n_pix:
            got_nside = int(round(math.sqrt(x.shape[-1] / 12)))
            raise ValueError(
                f"'{name}' has {x.shape[-1]} pixels (nside≈{got_nside}), "
                f"but this HEALPixSHT was built for nside={self.nside} "
                f"({self.n_pix} pixels).  "
                f"Create a new HEALPixSHT(level={int(round(math.log2(got_nside)))}) or pass the "
                f"correct map."
            )

    def _check_alm(self, x: torch.Tensor, name: str = "alm") -> None:
        """Raise a clear error if the last dimension doesn't match n_alm."""
        if x.shape[-1] != self.n_alm:
            raise ValueError(
                f"'{name}' has {x.shape[-1]} coefficients, "
                f"but this HEALPixSHT expects n_alm={self.n_alm} "
                f"(nside={self.nside}, lmax={self.lmax})."
            )

    # ------------------------------------------------------------------
    # Spin-0 analysis: map2alm
    # ------------------------------------------------------------------

    def map2alm(
        self,
        im:   ArrayLike,
        nest: bool = False,
    ) -> torch.Tensor:
        """
        Spin-0 analysis: HEALPix map → spherical harmonic coefficients.

            a_lm = ∫ f(θ,φ) · Y_lm*(θ,φ) dΩ

        Fully differentiable with respect to ``im``.

        Parameters
        ----------
        im   : (..., N) real array-like / Tensor.  RING order by default.
        nest : bool  Set True for NESTED input.

        Returns
        -------
        alm : (..., K) complex Tensor.
              K = (lmax+1)·(lmax+2)//2.
              Flat layout: ``[m=0: l=0..lmax | m=1: l=1..lmax | … | m=lmax]``.
        """
        im      = self._to_real(im)
        self._check_map(im, "im")
        leading = im.shape[:-1]
        B       = max(1, int(np.prod(leading)))
        im_2d   = im.reshape(B, self.n_pix)

        if nest:
            # ``_ring2nest_np[r]`` is the NESTED destination of RING
            # sample ``r``.  Gathering a NESTED input in RING order must
            # therefore use that forward map, not its inverse.
            im_2d = im_2d[:, self._r2n()]   # nested → ring

        lmax = self.lmax
        K    = self.n_alm
        ft   = self._rings_to_fft(im_2d)   # (B, R, lmax+1) complex

        alm = torch.zeros(B, K, dtype=self.cdtype, device=self.device)

        for m in range(lmax + 1):
            plm = self._plm_ana[m]              # (L, R) complex
            fm  = ft[:, :, m]                  # (B, R) complex

            # a_lm = Σ_r plm[l-m, r] · ft[r, m]
            tmp = torch.einsum("lr,br->bl", plm, fm)   # (B, L)
            alm[:, self._alm_slice(m)] = tmp

        return alm.reshape(leading + (K,))

    # ------------------------------------------------------------------
    # Spin-0 synthesis: alm2map
    # ------------------------------------------------------------------

    def alm2map(
        self,
        alm:  ArrayLike,
        nest: bool = False,
    ) -> torch.Tensor:
        """
        Spin-0 synthesis: a_lm → HEALPix map.

            f(θ,φ) = Σ_{lm} a_lm · Y_lm(θ,φ)

        Fully differentiable with respect to ``alm``.

        Parameters
        ----------
        alm  : (..., K) complex array-like / Tensor.
        nest : bool  Set True for NESTED output.

        Returns
        -------
        im : (..., N) real Tensor.
        """
        alm     = self._to_complex(alm)
        self._check_alm(alm, "alm")
        leading = alm.shape[:-1]
        B       = max(1, int(np.prod(leading)))
        alm_2d  = alm.reshape(B, self.n_alm)

        lmax   = self.lmax
        R      = self.n_rings
        ft_syn = torch.zeros(B, R, lmax + 1, dtype=self.cdtype, device=self.device)

        for m in range(lmax + 1):
            if not self._plm_syn:
                self._build_legendre_tables(synthesis=True)
            plm = self._plm_syn[m]                        # (L, R) complex
            a   = alm_2d[:, self._alm_slice(m)]           # (B, L) complex

            # F^r_m = Σ_l a_lm · Y_lm(θ_r, 0) = Σ_l a_lm · plm_syn[l-m, r]
            ft_syn[:, :, m] = torch.einsum("bl,lr->br", a, plm)  # (B, R)

        im_2d = self._fft_to_rings(ft_syn)   # (B, N) real

        if nest:
            # Output position ``n`` needs the RING sample given by
            # ``_nest2ring_np[n]``.
            im_2d = im_2d[:, self._n2r()]    # ring → nested

        return im_2d.reshape(leading + (self.n_pix,))

    # ------------------------------------------------------------------
    # Spin-s analysis: map2alm_spin
    # ------------------------------------------------------------------

    def map2alm_spin(
        self,
        Q:    ArrayLike,
        U:    ArrayLike,
        spin: int  = 2,
        nest: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spin-s analysis: (Q, U) → E-mode and B-mode alm tensors.

        For spin=2:
            Q = +P cos(2χ),  U = +P sin(2χ)  (IAU CMB convention)
        For spin=1:
            Q = longitudinal (div) component,  U = transverse (curl) component
        For spin=0:
            Delegates to map2alm(Q) and map2alm(U) independently.

        The E/B decomposition follows:
            Σ+(m) = Σ_r  +sY_lm*(θ_r) · F_r^+(m),   F^+ = FFT(Q + iU)
            Σ-(m) = Σ_r  -sY_lm*(θ_r) · F_r^-(m),   F^- = FFT(Q - iU)
            almE  = -(Σ+ + Σ-) / 2
            almB  =  (Σ+ - Σ-) / (2i)

        Fully differentiable with respect to Q and U.

        Parameters
        ----------
        Q, U : (..., N) real Tensors.  RING order by default.
        spin : int  Spin weight (1 or 2; use 0 for scalar pair).
        nest : bool  Set True for NESTED input.

        Returns
        -------
        almE, almB : (..., K) complex Tensors.
        """
        if spin == 0:
            # sign_E = (-1)^(1-0//2) = -1,  sign_B = (-1)^(0+1) = -1
            return -self.map2alm(Q, nest=nest), -self.map2alm(U, nest=nest)

        self._ensure_spin(spin)

        Q = self._to_real(Q);  U = self._to_real(U)
        self._check_map(Q, "Q");  self._check_map(U, "U")
        leading = Q.shape[:-1]
        B       = max(1, int(np.prod(leading)))
        Q2      = Q.reshape(B, self.n_pix)
        U2      = U.reshape(B, self.n_pix)

        if nest:
            Q2 = Q2[:, self._r2n()]          # nested → ring
            U2 = U2[:, self._r2n()]

        # Complex spin combinations
        QpU = torch.complex(Q2, U2)    # Q + iU
        QmU = torch.complex(Q2, -U2)  # Q - iU

        lmax = self.lmax
        K    = self.n_alm

        ft_p = self._rings_to_fft(QpU)   # (B, R, lmax+1)
        ft_m = self._rings_to_fft(QmU)   # (B, R, lmax+1)

        almE = torch.zeros(B, K, dtype=self.cdtype, device=self.device)
        almB = torch.zeros(B, K, dtype=self.cdtype, device=self.device)

        Yp = self._spin_Yp[spin]
        Ym = self._spin_Ym[spin]

        # E/B decomposition compatible with healpy/libsharp for any spin s:
        #
        #   almE = sign_E · (-(sg·Σ+ + Σ-)) / 2
        #   almB = sign_B · ( (sg·Σ+ - Σ-)) / (2i)
        #
        # where:
        #   sg     = (-1)^spin
        #   sign_E = (-1)^(1 - spin//2)   → -1, -1, +1  for spin = 0, 1, 2
        #   sign_B = (-1)^(spin+1)         → -1, +1, -1  for spin = 0, 1, 2
        #
        # These sign factors are determined empirically by comparing against
        # healpy.map2alm_spin for spin=0,1,2 and are a consequence of the
        # sign convention difference between the `spherical` Wigner package
        # (Boyle) and libsharp (healpy).

        sg     = (-1) ** spin
        sign_E = (-1) ** (1 - spin // 2)
        sign_B = (-1) ** (spin + 1)

        for m in range(lmax + 1):
            yp  = Yp[m]
            ym  = Ym[m]
            fpm = ft_p[:, :, m]
            fmm = ft_m[:, :, m]

            Sp = torch.einsum("lr,br->bl", yp, fpm)   # (B, L)
            Sm = torch.einsum("lr,br->bl", ym, fmm)   # (B, L)

            almE[:, self._alm_slice(m)] = sign_E * (-(sg * Sp + Sm)) / 2.0
            almB[:, self._alm_slice(m)] = sign_B *  ( (sg * Sp - Sm)) / (2.0j)

        return (
            almE.reshape(leading + (K,)),
            almB.reshape(leading + (K,)),
        )

    # ------------------------------------------------------------------
    # Spin-s synthesis: alm2map_spin
    # ------------------------------------------------------------------

    def alm2map_spin(
        self,
        almE: ArrayLike,
        almB: ArrayLike,
        spin: int  = 2,
        nest: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spin-s synthesis: (almE, almB) → (Q, U) maps.

        This is the adjoint of map2alm_spin.

        For spin=2, the reconstructed maps are:
            Q(n) = Re[ Σ_{lm} (-almE + i·almB) · +2Y_lm(n) ]
            U(n) = Im[ Σ_{lm} (-almE + i·almB) · +2Y_lm(n) ]

        For spin=0: delegates to alm2map(almE) and alm2map(almB).

        Fully differentiable with respect to almE and almB.

        Parameters
        ----------
        almE, almB : (..., K) complex Tensors.
        spin       : int  Spin weight.
        nest       : bool  Set True for NESTED output.

        Returns
        -------
        Q, U : (..., N) real Tensors.
        """
        if spin == 0:
            # adjoint of map2alm_spin spin=0: signs are -1,-1 so invert here too
            return self.alm2map(-almE, nest=nest), self.alm2map(-almB, nest=nest)

        self._ensure_spin(spin)

        almE = self._to_complex(almE)
        almB = self._to_complex(almB)
        self._check_alm(almE, "almE");  self._check_alm(almB, "almB")
        leading = almE.shape[:-1]
        B       = max(1, int(np.prod(leading)))
        E2      = almE.reshape(B, self.n_alm)
        B2      = almB.reshape(B, self.n_alm)

        lmax  = self.lmax
        R     = self.n_rings
        Yp    = self._spin_Yp_syn[spin]
        Ym    = self._spin_Ym_syn[spin]

        # Adjoint of the corrected map2alm_spin.
        #
        # Analysis:
        #   almE = sign_E · (-(sg·Σ+ + Σ-)) / 2
        #   almB = sign_B · (  (sg·Σ+ - Σ-)) / (2i)
        #
        # Solving for Σ+ and Σ-:
        #   -(sg·Σ+ + Σ-) / 2 = almE / sign_E  →  sg·Σ+ + Σ- = -2·almE·sign_E
        #    (sg·Σ+ - Σ-) / (2i) = almB / sign_B → sg·Σ+ - Σ- = 2i·almB·sign_B
        #
        # Adding:   2·sg·Σ+ = -2·almE·sign_E + 2i·almB·sign_B
        #   → Σ+ = sg · (-almE·sign_E + i·almB·sign_B)
        # Subtracting: 2·Σ- = -2·almE·sign_E - 2i·almB·sign_B
        #   → Σ- = -almE·sign_E - i·almB·sign_B
        #
        # Synthesis coefficients:
        #   coeff_p = sg · (-almE·sign_E + i·almB·sign_B)
        #   coeff_m =      (-almE·sign_E - i·almB·sign_B)

        sg     = (-1) ** spin
        sign_E = (-1) ** (1 - spin // 2)
        sign_B = (-1) ** (spin + 1)

        ft_p = torch.zeros(B, R, lmax + 1, dtype=self.cdtype, device=self.device)
        ft_m = torch.zeros(B, R, lmax + 1, dtype=self.cdtype, device=self.device)

        for m in range(lmax + 1):
            yp = Yp[m]
            ym = Ym[m]
            sl = self._alm_slice(m)
            eE = E2[:, sl];  eB = B2[:, sl]

            coeff_p = sg * (-eE * sign_E + 1j * eB * sign_B)   # (B, L)
            coeff_m =      (-eE * sign_E - 1j * eB * sign_B)   # (B, L)

            ft_p[:, :, m] = torch.einsum("bl,lr->br", coeff_p, yp)
            ft_m[:, :, m] = torch.einsum("bl,lr->br", coeff_m, ym)

        # Per-ring IFFT
        # ft_p synthesises (Q+iU),  ft_m synthesises (Q-iU)
        # → Q = Re[(QpU + QmU) / 2]
        # → U = Re[(QpU - QmU) / (2i)] = Im[(QpU - QmU) / 2] ... but simpler:
        # → Q = (QpU.real + QmU.real) / 2
        # → U = (QpU.imag - QmU.imag) / 2  ... actually:
        # QpU = Q + iU,  QmU = Q - iU  → Q = Re(QpU), U = Im(QpU)  only if exact
        # Using both for numerical stability:
        #   Q = (QpU + QmU).real / 2
        #   U = (QpU - QmU).imag / 2
        QpU = self._fft_to_rings_complex(ft_p)   # (B, N) ≈ Q + iU
        QmU = self._fft_to_rings_complex(ft_m)   # (B, N) ≈ Q - iU
        Q2  = ((QpU + QmU) / 2.0).real
        U2  = ((QpU - QmU) / 2.0j).real

        if nest:
            Q2 = Q2[:, self._n2r()]          # ring → nested
            U2 = U2[:, self._n2r()]

        return (
            Q2.reshape(leading + (self.n_pix,)),
            U2.reshape(leading + (self.n_pix,)),
        )

    def _fft_to_rings_complex(self, ft_syn: torch.Tensor) -> torch.Tensor:
        """
        Per-ring IFFT returning complex output (needed for spin-s synthesis).

        For a COMPLEX field (Q+iU), the synthesis at pixel j of ring r is:

            (Q+iU)_j = Σ_{m=0}^{lmax} G_m · exp(i·2π·m·j/Nr)

        where G_m = ft_up[r, m].  Unlike the real case (_fft_to_rings),
        there is NO conjugate symmetry and NO factor-2 for m>0.
        We simply fold the lmax+1 modes into a Nr-point spectrum by aliasing:

            H[m % Nr] += Nr · G_m    (no conjugate term)

        then IFFT(H)[j] = (Q+iU)_j.
        """
        lmax = self.lmax
        B    = ft_syn.shape[0]
        dev  = ft_syn.device

        ft_up = ft_syn * self._phase[None].conj()   # (B, R, lmax+1)

        rings: List[torch.Tensor] = []
        for r in range(self.n_rings):
            Nr = int(self._ring_n[r])
            H  = torch.zeros(B, Nr, dtype=self.cdtype, device=dev)
            # m=0: weight 1
            H[:, 0] = H[:, 0] + Nr * ft_up[:, r, 0]
            # m>0: weight 2 (positive-freq only spectrum, factor 2 for m>0)
            for m in range(1, lmax + 1):
                H[:, m % Nr] = H[:, m % Nr] + 2 * Nr * ft_up[:, r, m]
            rings.append(torch.fft.ifft(H, dim=-1))   # (B, Nr) complex

        return torch.cat(rings, dim=1)   # (B, N) complex

    # ------------------------------------------------------------------
    # Power spectrum: anafast
    # ------------------------------------------------------------------

    def anafast(
        self,
        im:   ArrayLike,
        map2: Optional[ArrayLike] = None,
        spin: int  = 0,
        nest: bool = False,
    ) -> torch.Tensor:
        """
        Angular power spectrum C_l (or cross-spectrum C_l^{12}).

        For spin=0 (scalar):
            ``C_l = 1/(2l+1) × [ |a_l0|² + 2 Σ_{m=1}^{l} |a_lm|² ]``

        For spin>0 (E/B modes):
            C_l^{EE}, C_l^{BB}, C_l^{EB} are returned as a (3, lmax+1) tensor.

        Fully differentiable with respect to im (and map2).

        Parameters
        ----------
        im   : (..., N) or (..., 2, N) array-like.
               Scalar map, or [Q, U] stacked on axis -2 for spin > 0.
        map2 : same shape as im or None.  If given, compute cross-spectrum.
        spin : int  0 for scalar, 1 or 2 for polarisation.
        nest : bool

        Returns
        -------
        cl : (..., lmax+1) float Tensor for spin=0.
             (..., 3, lmax+1) float Tensor for spin>0  — [EE, BB, EB].
        """
        lmax = self.lmax

        if spin == 0:
            # ---------- scalar ----------
            alm1 = self.map2alm(im, nest=nest)
            alm2 = self.map2alm(map2, nest=nest) if map2 is not None else alm1

            leading = alm1.shape[:-1]
            B       = max(1, int(np.prod(leading)))
            a1      = alm1.reshape(B, self.n_alm)
            a2      = alm2.reshape(B, self.n_alm)

            cl = torch.zeros(B, lmax + 1, dtype=self.dtype, device=self.device)

            for m in range(lmax + 1):
                sl      = self._alm_slice(m)
                power   = (a1[:, sl] * a2[:, sl].conj()).real   # (B, L)
                w       = 1.0 if m == 0 else 2.0
                L       = lmax - m + 1
                l_idx   = torch.arange(m, lmax + 1, device=self.device)
                two_lp1 = (2.0 * l_idx.float() + 1.0)          # (L,)
                cl[:, m : m + L] += w * power / two_lp1[None]

            return cl.reshape(leading + (lmax + 1,))

        else:
            # ---------- spin-s : E/B modes ----------
            im_t = self._to_real(im) if not torch.is_tensor(im) else im.to(self.device)
            if im_t.shape[-2] != 2:
                raise ValueError(
                    "For spin>0, im must have shape (..., 2, N) with "
                    "im[..., 0, :] = Q and im[..., 1, :] = U."
                )
            Q1 = im_t[..., 0, :]
            U1 = im_t[..., 1, :]

            almE1, almB1 = self.map2alm_spin(Q1, U1, spin=spin, nest=nest)

            if map2 is not None:
                m2   = self._to_real(map2) if not torch.is_tensor(map2) else map2.to(self.device)
                Q2   = m2[..., 0, :]
                U2   = m2[..., 1, :]
                almE2, almB2 = self.map2alm_spin(Q2, U2, spin=spin, nest=nest)
            else:
                almE2, almB2 = almE1, almB1

            leading = almE1.shape[:-1]
            B       = max(1, int(np.prod(leading)))
            E1 = almE1.reshape(B, self.n_alm)
            B1 = almB1.reshape(B, self.n_alm)
            E2 = almE2.reshape(B, self.n_alm)
            B2 = almB2.reshape(B, self.n_alm)

            clEE = torch.zeros(B, lmax + 1, dtype=self.dtype, device=self.device)
            clBB = torch.zeros_like(clEE)
            clEB = torch.zeros_like(clEE)

            for m in range(lmax + 1):
                sl       = self._alm_slice(m)
                w        = 1.0 if m == 0 else 2.0
                l_idx    = torch.arange(m, lmax + 1, device=self.device)
                two_lp1  = (2.0 * l_idx.float() + 1.0)

                pEE = (E1[:, sl] * E2[:, sl].conj()).real
                pBB = (B1[:, sl] * B2[:, sl].conj()).real
                pEB = (E1[:, sl] * B2[:, sl].conj()).real

                clEE[:, m : m + (lmax - m + 1)] += w * pEE / two_lp1[None]
                clBB[:, m : m + (lmax - m + 1)] += w * pBB / two_lp1[None]
                clEB[:, m : m + (lmax - m + 1)] += w * pEB / two_lp1[None]

            cl3 = torch.stack([clEE, clBB, clEB], dim=1)  # (B, 3, lmax+1)
            return cl3.reshape(leading + (3, lmax + 1))

    # ------------------------------------------------------------------
    # Convenience: curl / divergence  (spin=1)
    # ------------------------------------------------------------------

    def uv_to_curl_div(
        self,
        u: ArrayLike,
        v: ArrayLike,
        nest: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose a tangent-plane vector field (u, v) into curl and divergence.

        Uses spin=1 spherical harmonics:
            almE (divergence-free part)  →  div map
            almB (curl-free part)        →  curl map

        Parameters
        ----------
        u, v : (..., N) real arrays — east and north components.

        Returns
        -------
        div, curl : (..., N) real Tensors.
        """
        almE, almB = self.map2alm_spin(-v, u, spin=1, nest=nest)
        div  = self.alm2map(almE, nest=nest)
        curl  = self.alm2map(almB, nest=nest)
        return div, curl

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HEALPixSHT(level={self.level}, nside={self.nside}, lmax={self.lmax}, "
            f"n_rings={self.n_rings}, n_alm={self.n_alm}, "
            f"dtype={self.dtype}, device={self.device})"
        )
