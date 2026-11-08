"""A PySM dust sky as the PR3 maps of prepare_planck.py: a known sky for the validation.

    python make_pysm_sky.py OUT_DIR [--model d12 --nside-in 2048 --nside 256 --fwhm 4.94]

Evaluates the PySM 3 model at 353 GHz (I, Q, U, in uK_CMB) and 857 GHz (I, in
MJy/sr, as the PR3 857 GHz map), at nside-in, smooths it with the 353 GHz beam
(Gaussian, --fwhm arcmin) and writes NESTED parent means at each --nside:
OUT_DIR/sky353_{I,Q,U}_n{nside}.npy and sky857_I_n{nside}.npy. Delta bandpasses.
"""

import argparse
import os

import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u


def to_nested_means(m, nside):
    m = hp.reorder(m, r2n=True)
    while m.size > 12 * nside * nside:
        m = m.reshape(-1, 4).mean(1)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--model", default="d12")
    p.add_argument("--nside-in", type=int, default=2048)
    p.add_argument("--nside", type=int, nargs="+", default=[256])
    p.add_argument("--fwhm", type=float, default=4.94, help="arcmin, Planck 353 GHz")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sky = pysm3.Sky(nside=a.nside_in, preset_strings=[a.model])
    for freq, unit, fields, name in ((353, u.uK_CMB, "IQU", "sky353"), (857, u.MJy / u.sr, "I", "sky857")):
        m = sky.get_emission(freq * u.GHz).to(unit, equivalencies=u.cmb_equivalencies(freq * u.GHz)).value
        m = hp.smoothing(m[: len(fields)] if len(fields) > 1 else m[0], fwhm=np.radians(a.fwhm / 60),
                         pol=len(fields) > 1, iter=1)
        for i, f in enumerate(fields):
            mf = m[i] if len(fields) > 1 else m
            for nside in a.nside:
                out = to_nested_means(mf, nside)
                np.save(os.path.join(a.out, f"{name}_{f}_n{nside}.npy"), out)
                print(f"{a.model} {freq} GHz {f} nside {nside}: std {out.std():.4g}", flush=True)


if __name__ == "__main__":
    main()
