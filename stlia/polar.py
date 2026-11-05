"""E and B maps of a full-sky (Q, U) pair, differentiable.

The scattering statistics of Q and U taken separately depend on the frame in
which Q and U are defined; those of the E and B maps do not.  FOSCAT computes
its spin-2 statistics with wavelets built from spin-2 harmonics, which give
the wavelet coefficients of E and B directly from (Q, U); on the full sky this
is the scalar scattering of the E and B maps themselves, which is what
`EBTransform` provides to the scalar engine:

    (Q, U) --map2alm_spin(spin 2)--> (a_E, a_B) --alm2map--> (E, B)

with the transforms of `stlia.sht` (HEALPix convention, as healpy's
``map2alm(pol=True)`` and ``alm2map``).  The map is linear: a noise added to
(Q, U) is a noise added to (E, B), so the noise biases of the statistics
(``unbiased()`` of `train_denoise`) are the same.
"""

import torch

from .sht import HEALPixSHT


class EBTransform:
    """(Q, U) `[..., 2, npix]` NESTED -> (E, B) `[..., 2, npix]` NESTED,
    differentiable, to `lmax = 3 nside - 1`."""

    def __init__(self, nside, device=None, dtype=torch.float32):
        self.sht = HEALPixSHT(nside, dtype=dtype, device=device)

    def __call__(self, qu):
        a_e, a_b = self.sht.map2alm_spin(qu[..., 0, :], qu[..., 1, :], spin=2, nest=True)
        eb = torch.stack([self.sht.alm2map(a_e, nest=True), self.sht.alm2map(a_b, nest=True)], -2)
        return eb.to(qu.dtype)

    def many(self, q, u, chunk=8):
        """(E, B) of the pairs `q`, `u` `[k, npix]`, without gradient -> two `[k, npix]`."""
        out = []
        with torch.no_grad():
            for k in range(0, len(q), chunk):
                out.append(self(torch.stack([q[k:k + chunk], u[k:k + chunk]], -2)))
        eb = torch.cat(out)
        return eb[:, 0], eb[:, 1]
