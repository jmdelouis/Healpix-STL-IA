#!/bin/bash
# Degrade the downloaded PR3 maps and FFP10 simulations (scripts/download_planck.sh)
# to nside 1024 and 512 (NESTED .npy in uK_CMB, parent means), as prepare_data.slurm
# does for 256/128/64, into $1 (default $R/data/maps_hi): about 60 GB.
#
#   sbatch $CPU_OPTS -o $RUNS/slurm.%j.log scripts/prepare_data_hires.slurm [OUT_DIR]
#
# Train with --data OUT_DIR (it holds every file the driver reads at these nsides).
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
P=$R/data/planck
M=${1:-$R/data/maps_hi}
S=$R/src/scripts
N="--nside 1024 512"
set -e
mkdir -p $M
$PY $S/prepare_planck.py $P/HFI_SkyMap_353-psb_2048_R3.01_full.fits $M --name planck353 $N
for h in 1 2; do
    $PY $S/prepare_planck.py $P/HFI_SkyMap_353-psb_2048_R3.01_halfmission-$h.fits $M --name planck353hm$h \
        --fill-from planck353 $N
done
$PY $S/prepare_planck.py $P/HFI_SkyMap_857_2048_R3.01_full.fits $M --name planck857 --fields I --scale 1 $N
$PY $S/noise_std.py $M $(ls $P/noise/ffp10_noise_353_psb_full_map_mc_000??.fits | sort) $N | grep -v done
for h in 1 2; do
    $PY $S/noise_std.py $M $(ls $P/noise_hm/ffp10_noise_353_psb_hm${h}_map_mc_000??.fits | sort) --name noisehm$h $N \
        | grep -v done
done
ls $M | wc -l
du -sh $M
