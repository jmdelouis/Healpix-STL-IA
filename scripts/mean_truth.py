"""Mean over the noise simulations mc1-9 of the eval_truth.py lines.   python scripts/mean_truth.py RUN/eval_truth.txt ..."""
import re
import sys

import numpy as np

for fn in sys.argv[1:]:
    acc, bands, cur = {}, [], None
    for line in open(fn).read().splitlines():
        m = re.match(r"u_mc\d\.npz mc\d (.*)", line)
        if m:
            rest = m.group(1)
            if rest.startswith("u-s rms"):
                bands.append([float(x) for x in rest.split(":")[1].split()])
                cur = None
            elif rest.startswith("|b|"):
                cur = rest.split(" ell")[0]
            else:
                cur = None
            continue
        if cur and line.startswith("    "):
            for part in line.split("|"):
                part = part.strip()
                name = re.sub(r"(\s-?\d+\.\d+)+$", "", part)
                acc.setdefault((cur, name), []).append([float(v) for v in re.findall(r"\s(-?\d+\.\d+)", " " + part[len(name):])])
    b = np.array(bands)
    print("==", fn, len(bands), "sims; u-s rms/noise by |b|:", " ".join(f"{x:.3f}" for x in b.mean(0)),
          f"(max dev {np.abs(b - b.mean(0)).max():.3f})")
    for (mask, name), v in acc.items():
        if name.startswith(("C(d-s)", "transfer")):
            continue
        a = np.array(v)
        print(f"{mask:14s} {name:12s}", " ".join(f"{x:.2f}" for x in a.mean(0)), f" sd<={a.std(0).max():.2f}")
