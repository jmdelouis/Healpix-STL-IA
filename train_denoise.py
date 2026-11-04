"""Command-line entry of :mod:`stlia.train_denoise` (kept at the top level for the
recipe of REPRODUCE.md; the installed package also provides
`stlia-train-denoise`)."""

import sys

from stlia.train_denoise import main

if __name__ == "__main__":
    sys.exit(main())
