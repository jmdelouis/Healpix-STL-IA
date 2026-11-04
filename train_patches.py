"""Command-line entry of :mod:`stlia.train_patches` (kept at the top level for the
recipe of REPRODUCE.md; the installed package also provides
`stlia-train-patches`)."""

import sys

from stlia.train_patches import main

if __name__ == "__main__":
    sys.exit(main())
