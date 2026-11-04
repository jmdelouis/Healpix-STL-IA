#!/bin/bash
# One train_patches.py run with the CPU disc engine (32 cores, 128 GB).
#
#   sbatch $CPU_OPTS -o $RUNS/slurm.%j.log scripts/train_cpu.slurm RUN [train_patches.py options]
#
# or directly:  bash scripts/train_cpu.slurm RUN [options]
# The run goes to $RUNS/RUN (log, model.pt, train.json, eval_*.npz).
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
name=$1; shift
mkdir -p $RUNS/$name
cd $R/src
$PY train_patches.py --data $R/data/maps --out $RUNS/$name --device cpu --threads ${SLURM_CPUS_PER_TASK:-$(nproc)} \
    "$@" > $RUNS/$name/log 2>&1
