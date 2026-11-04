#!/bin/bash -l
# One patch-training run on a CPU node of the hpc cpu partition (the disc
# statistics of PatchEngine run on host threads).
#
#   sbatch -J NAME -c 32 scripts/train_cpu.slurm RUN_NAME [train_patches.py options]
#
# The run goes to $R/runs/RUN_NAME (log, model.pt, train.json, eval_*.npz).
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/jmdeloui/healpix-stl-ia/runs/slurm.%j.log
R=/scratch/jmdeloui/healpix-stl-ia
source $R/src/scripts/hpc_env.sh
name=$1; shift
mkdir -p $R/runs/$name
cd $R/src
$PY train_patches.py --data $R/data/maps --out $R/runs/$name --device cpu --threads $SLURM_CPUS_PER_TASK "$@" > $R/runs/$name/log 2>&1
