# Environment of the hpc (Datarmor) runs (bash):  source scripts/hpc_env.sh
#
# R holds src/ (this repository), $STL/src (STL-Dev-MPI sources: crates,
# python/stl_ensemble), $STL/target (stl-bridge built with the gpu feature)
# and data/maps (planck353_*_n*.npy).  STL: stl_dc (branch
# claude/disc-cross-stats: PatchEngine and cross statistics, the default) or
# stl (branch claude/amortized-synthesis, full-sky auto statistics only).
R=${R:-/scratch/jmdeloui/healpix-stl-ia}
STL=${STL:-stl_dc}
export G=$R/$STL
source $G/src/tools/hpc/env_gpu.sh
export STL_NATIVE_LIBRARY=$G/target/release/libstl_bridge.so
export PYTHONPATH=$R/src:$G/src/python/stl_ensemble
PY=${PY:-/scratch/jmdeloui/stl-bench128/env/bin/python}
