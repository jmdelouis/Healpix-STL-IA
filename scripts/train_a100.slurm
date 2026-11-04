#!/bin/bash -l
# One training run on one A100 of the hpc gpu partition.
#
#   sbatch -J NAME scripts/train_a100.slurm RUN_NAME [train_denoise.py options]
#
# The run goes to $R/runs/RUN_NAME (log, model.pt, train.json, eval_*.npz).
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/jmdeloui/healpix-stl-ia/runs/slurm.%j.log
R=/scratch/jmdeloui/healpix-stl-ia
source $R/src/scripts/hpc_env.sh
name=$1; shift
mkdir -p $R/runs/$name
cd $R/src
$PY train_denoise.py --data $R/data/maps --out $R/runs/$name --device 0 "$@" > $R/runs/$name/log 2>&1
