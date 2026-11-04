# Toolchain of the Ifremer Datarmor cluster: Rust (rustup under $W) and the
# CUDA 13.0 module (driver API, NVRTC).  Sourced through SITE_ENV.
W=${W:-$HOME/stl-mpi}
export RUSTUP_HOME=$W/rustup CARGO_HOME=$W/cargo
source $W/cargo/env
source /usr/share/Modules/init/bash 2>/dev/null || true
module load cuda/13.0.0
