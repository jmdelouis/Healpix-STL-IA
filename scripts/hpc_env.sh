# Environment of the hpc (Datarmor) runs (bash):  source scripts/hpc_env.sh
#
# R holds src/ (this repository), stl/src (STL-Dev-MPI sources: crates,
# python/stl_ensemble), stl/target (stl-bridge built with the gpu feature,
# build_bridge.slurm) and data/maps (planck353_*_n*.npy).
R=${R:-/scratch/jmdeloui/healpix-stl-ia}
export G=$R/stl
source $G/src/tools/hpc/env_gpu.sh
export STL_NATIVE_LIBRARY=$G/target/release/libstl_bridge.so
export PYTHONPATH=$R/src:$G/src/python/stl_ensemble
PY=${PY:-/scratch/jmdeloui/stl-bench128/env/bin/python}
