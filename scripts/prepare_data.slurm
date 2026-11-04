#!/bin/bash
# Degrade the downloaded PR3 maps and FFP10 simulations (scripts/download_planck.sh)
# to nside 256/128/64 (NESTED .npy in uK_CMB, parent means) into $R/data/maps.
# About 3 h on 4 cores, 24 GB of memory, 15 GB on disk.
#
#   sbatch $CPU_OPTS -o $RUNS/slurm.%j.log scripts/prepare_data.slurm    (or: bash scripts/prepare_data.slurm)
#
# The per-pixel std of `--noise pr3white` (nside 64 runs) is that of the
# simulations mc 0-9: noise_std.py is run on the 50 simulations (every
# degraded draw), then again on the first ten (the std).
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
P=$R/data/planck
M=$R/data/maps
S=$R/src/scripts
set -e
$PY $S/prepare_planck.py $P/HFI_SkyMap_353-psb_2048_R3.01_full.fits $M --name planck353
for h in 1 2; do
    $PY $S/prepare_planck.py $P/HFI_SkyMap_353-psb_2048_R3.01_halfmission-$h.fits $M --name planck353hm$h \
        --fill-from planck353
done
$PY $S/prepare_planck.py $P/HFI_SkyMap_857_2048_R3.01_full.fits $M --name planck857 --fields I --scale 1
$PY $S/noise_std.py $M $(ls $P/noise/ffp10_noise_353_psb_full_map_mc_000??.fits | sort) | grep -v done
for h in 1 2; do
    $PY $S/noise_std.py $M $(ls $P/noise_hm/ffp10_noise_353_psb_hm${h}_map_mc_000??.fits | sort) --name noisehm$h \
        | grep -v done
done
$PY $S/noise_std.py $M $(ls $P/noise/ffp10_noise_353_psb_full_map_mc_0000?.fits | sort) | grep -v done
ls $M | wc -l
