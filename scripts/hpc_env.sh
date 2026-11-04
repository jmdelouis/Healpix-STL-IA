# Site settings of the runs of the paper: Ifremer Datarmor ("hpc"), Slurm,
# RHEL 9, nodes of 8 x A100 80 GB (partition gpu) and CPU nodes (partition
# cpu); Rust 1.99.1 installed with rustup under $W, CUDA 13.0 module.
#
#   source scripts/hpc_env.sh      (bash; sets the variables of scripts/env.sh)
#
# For another machine, write the same few lines for it (or export them by
# hand) and source scripts/env.sh.
export R=${R:-/scratch/$USER/healpix-stl-ia}
export STL=${STL:-stl_dc}
export SITE_ENV=${SITE_ENV:-$R/src/scripts/site/datarmor.sh}
export CUDARC_CUDA_VERSION=${CUDARC_CUDA_VERSION:-13000}
export PY=${PY:-/scratch/$USER/stl-bench128/env/bin/python}
export GPU_OPTS=${GPU_OPTS:-"-p gpu --gres=gpu:a100:1 -c 6 --mem=48G -t 12:00:00"}
export CPU_OPTS=${CPU_OPTS:-"-p cpu -c 32 --mem=128G -t 24:00:00"}
source $R/src/scripts/env.sh
