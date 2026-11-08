#!/bin/bash
# Time and memory of one frozen-bias training step (data, l1, l2, l3), native
# U-Net of STL-Dev-MPI (NativeUNet, commit b580210, built in
# $R/$STL) against the torch pipeline of train_denoise.py, on one GPU:
#
#   STL=stl_nu sbatch $GPU_OPTS -o $RUNS/slurm.%j.log scripts/bench_native.slurm
#
# Writes $RUNS/bench_native/{native,torch}_n{64,256}.json and log.
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
out=$RUNS/bench_native
mkdir -p $out
bench=$G/src/python/stl_ensemble/bench_native_unet.py
cd $R/src
for cfg in "64 4" "256 6"; do
    set -- $cfg
    for impl in native torch; do
        # the last line of stdout is the JSON summary
        $PY $bench --impl $impl --nside $1 --levels $2 --steps 5 2>> $out/log | tail -1 > $out/${impl}_n$1.json
    done
done
