"""The Planck PR3 GNILC 353 GHz dust polarisation map, as the PR3 maps of prepare_planck.py.

    python prepare_gnilc.py COM_CompMap_IQU-thermaldust-gnilc-varres_2048_R3.00.fits OUT_DIR [--nside 256]

Writes OUT_DIR/gnilc353_{Q,U}_n{nside}.npy: NESTED parent means of Q and U, in
uK_CMB. GNILC is used only as a reference in the figures (make_figures_real.py --gnilc).
"""

import argparse
import os

import healpy as hp
import numpy as np
from astropy.io import fits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fits")
    p.add_argument("out")
    p.add_argument("--nside", type=int, default=256)
    a = p.parse_args()
    head = fits.getheader(a.fits, 1)
    names = [head.get(f"TTYPE{i}", "") for i in range(1, head["TFIELDS"] + 1)]
    units = [head.get(f"TUNIT{i}", "") for i in range(1, head["TFIELDS"] + 1)]
    print("columns", list(zip(names, units)), "ordering", head.get("ORDERING"), "nside", head.get("NSIDE"))
    for f in ("Q", "U"):
        i = next(k for k, n in enumerate(names) if n.upper().startswith(f))
        scale = {"K_CMB": 1e6, "MK_CMB": 1e3, "UK_CMB": 1.0}[units[i].upper().replace(" ", "")]
        m = hp.read_map(a.fits, field=i, nest=True) * scale
        m = hp.ud_grade(m, a.nside, order_in="NESTED", order_out="NESTED")
        np.save(os.path.join(a.out, f"gnilc353_{f}_n{a.nside}.npy"), m)
        print(f, f"rms {m.std():.4g} uK_CMB")


if __name__ == "__main__":
    main()
