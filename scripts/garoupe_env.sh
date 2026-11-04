# Environment of the garoupe test runs (bash):  source scripts/garoupe_env.sh
#
# G holds src/ (this repository), stl_ensemble/ (STL-Dev-MPI python/stl_ensemble,
# for stl_port.native), lib/libstl_bridge.so (stl-bridge built with the gpu
# feature) and data/ (planck353_*_n*.npy).
G=${G:-/tmp/jmdeloui-stlia}
source $HOME/stl-gpu-work/src/tools/garoupe/env.sh
export STL_NATIVE_LIBRARY=$G/lib/libstl_bridge.so
export PYTHONPATH=$G/src:$G/stl_ensemble
PY=${PY:-/tmp/jmdeloui-diffusion/venv/bin/python}
