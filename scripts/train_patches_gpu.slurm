#!/bin/bash
# One train_patches.py run with the disc engine on the GPU (STL-Dev-MPI 20329e1).
#
#   sbatch $GPU_OPTS -o $R/runs/slurm.%j.log scripts/train_patches_gpu.slurm RUN [train_patches.py options]
#
# or directly:  bash scripts/train_patches_gpu.slurm RUN [options]
# CACHE_BYTES: GPU budget of the disc geometries (default 6e10, for 80 GB;
# 3e9 on a 16 GB GPU).  The run goes to $R/runs/RUN.
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
name=$1; shift
mkdir -p $R/runs/$name
cd $R/src
$PY train_patches.py --data $R/data/maps --out $R/runs/$name --stl-backend gpu --device cuda:0 \
    --cache-bytes ${CACHE_BYTES:-6e10} --threads ${SLURM_CPUS_PER_TASK:-$(nproc)} "$@" > $R/runs/$name/log 2>&1
