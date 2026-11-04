#!/bin/bash
# Build the STL-Dev-MPI Rust bridge with the gpu feature into
# $R/$STL/target/release/libstl_bridge.so (STL_NATIVE_LIBRARY of env.sh).
#
#   git clone https://github.com/jmdelouis/STL-Dev-MPI $R/$STL/src
#   git -C $R/$STL/src checkout 20329e1d2aeb6b822e194f671367e9eed8eb8193
#   sbatch $CPU_OPTS -o $R/runs/slurm.%j.log scripts/build_bridge.slurm    (or: bash scripts/build_bridge.slurm)
#
# Needs cargo (Rust stable; built with 1.99.1) and CUDARC_CUDA_VERSION; no
# MPI and no libclang (stl-bridge depends on stl-core and stl-gpu only).
# The runs of Sects. 3-4 used 66975a4dead97b8828721b6fbb67b2147a3fed48, the
# disc engine on the GPU (train_patches.py --stl-backend gpu) needs 20329e1;
# 20329e1 is 66975a4 plus two commits: the same full-sky statistics; on discs,
# interpolation cells of weight <= 1e-9 are no longer read (rounding-level change).
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
: "${CUDARC_CUDA_VERSION:?set CUDARC_CUDA_VERSION (driver CUDA version, e.g. 12080)}"
cd $G/src && cargo build --release -p stl-bridge --features gpu 2>&1 | tail -5
ls -la $STL_NATIVE_LIBRARY
