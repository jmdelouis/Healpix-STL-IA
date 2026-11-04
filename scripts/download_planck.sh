#!/bin/bash
# Download the Planck PR3 maps and FFP10 noise simulations of the paper from
# the Planck Legacy Archive into $R/data/planck (about 60 GB).
#
#   R=/scratch/$USER/healpix-stl-ia bash scripts/download_planck.sh [NSIMS]
#
# NSIMS (default 50): simulations mc 0 .. NSIMS-1 of the full mission and of
# the two half missions.
set -e
R=${R:-/scratch/$USER/healpix-stl-ia}
N=${1:-50}
P=$R/data/planck
mkdir -p $P/noise $P/noise_hm
URL=http://pla.esac.esa.int/pla/aio/product-action
get() {  # get KEY FILE DIR
    [ -s "$3/$2" ] || curl -sSf -o "$3/$2" "$URL?$1=$2"
}
for f in HFI_SkyMap_353-psb_2048_R3.01_full.fits HFI_SkyMap_353-psb_2048_R3.01_halfmission-1.fits \
         HFI_SkyMap_353-psb_2048_R3.01_halfmission-2.fits HFI_SkyMap_857_2048_R3.01_full.fits; do
    get MAP.MAP_ID $f $P
done
for k in $(seq -f "%05g" 0 $((N - 1))); do
    get SIMULATED_MAP.FILE_ID ffp10_noise_353_psb_full_map_mc_$k.fits $P/noise
    for h in 1 2; do
        get SIMULATED_MAP.FILE_ID ffp10_noise_353_psb_hm${h}_map_mc_$k.fits $P/noise_hm
    done
done
ls $P $P/noise $P/noise_hm | wc -l
