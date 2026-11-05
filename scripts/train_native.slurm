#!/bin/bash
# One training of the native U-Net of STL-Dev-MPI (NativeUNet, branch
# claude/native-unet, built in $R/$STL) on the PR3 maps, with its driver
# train_native_unet.py (the options of train_denoise.py --real):
#
#   STL=stl_nu sbatch $GPU_OPTS -o $RUNS/slurm.%j.log scripts/train_native.slurm RUN [driver options]
#
# The run goes to $RUNS/RUN (log, model.pt, train.json, eval_*.npz), as train_gpu.slurm.
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
name=$1; shift
mkdir -p $RUNS/$name
cd $R/src
$PY $G/src/python/stl_ensemble/train_native_unet.py --data $R/data/maps --out $RUNS/$name "$@" > $RUNS/$name/log 2>&1
