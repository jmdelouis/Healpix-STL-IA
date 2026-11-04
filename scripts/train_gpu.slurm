#!/bin/bash
# One train_denoise.py run on one GPU.  With Slurm (from $R/src, after
# sourcing the environment, see REPRODUCE.md):
#
#   sbatch $GPU_OPTS -o $RUNS/slurm.%j.log scripts/train_gpu.slurm RUN [train_denoise.py options]
#
# or directly on a GPU machine:  bash scripts/train_gpu.slurm RUN [options]
# The run goes to $RUNS/RUN (log, model.pt, train.json, eval_*.npz).
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
name=$1; shift
mkdir -p $RUNS/$name
cd $R/src
$PY train_denoise.py --data $R/data/maps --out $RUNS/$name --device 0 "$@" > $RUNS/$name/log 2>&1
