# Environment of the build, the runs and the figures, on any Linux machine
# with an NVIDIA GPU (bash):
#
#   export R=/path/to/workdir [SITE_ENV=... PY=... CUDARC_CUDA_VERSION=...]
#   source $R/src/scripts/env.sh
#
# R (required): work directory, laid out as
#   $R/src          this repository
#   $R/$STL/src     the STL-Dev-MPI checkout (STL, default `stl`)
#   $R/$STL/target  its cargo build (libstl_bridge.so)
#   $R/data/planck  the PLA downloads, $R/data/maps the degraded maps
#   $RUNS           one directory per run (RUNS, default $R/runs: set it
#                   elsewhere when the work file system is full)
# SITE_ENV: a file sourced first, for what the site needs (modules, rustup,
#   the CUDA runtime); see scripts/hpc_env.sh for an example.
# PY: python 3.10+ with torch (CUDA build), numpy, scipy, healpy, matplotlib
#   (default: python3 of the PATH).
# CUDARC_CUDA_VERSION (build time): the CUDA version of the driver, as printed
#   by nvidia-smi, e.g. 11040 for 11.4, 12080 for 12.8, 13000 for 13.0.
# At run time the GPU engine compiles its kernels with NVRTC: libnvrtc must be
# on LD_LIBRARY_PATH (CUDA toolkit module, or the PyPI nvidia-cuda-nvrtc wheel).
: "${R:?set R to the work directory}"
export R
[ -n "$SITE_ENV" ] && source "$SITE_ENV"
export STL=${STL:-stl}
export RUNS=${RUNS:-$R/runs}
export G=$R/$STL
export CARGO_TARGET_DIR=$G/target
export STL_NATIVE_LIBRARY=$G/target/release/libstl_bridge.so  # follows STL, also when re-sourced in a job
export PYTHONPATH=$R/src:$G/src/python/stl_ensemble${PYTHONPATH:+:$PYTHONPATH}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$RUNS/.mpl}
export PY=${PY:-python3}
# Slurm options of the GPU (one GPU with >= 40 GB) and CPU (32 cores, 128 GB)
# jobs; empty when the jobs are run with bash on the current machine.
export GPU_OPTS=${GPU_OPTS:-}
export CPU_OPTS=${CPU_OPTS:-}
