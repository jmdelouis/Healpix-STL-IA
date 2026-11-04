#!/bin/bash
# Every run and figure of the paper (REPRODUCE.md).  After the environment
# (scripts/env.sh or a site file), the build and the data:
#
#   bash scripts/reproduce.sh runs       submit every run (Slurm), or run them
#                                        one after the other when GPU_OPTS is empty
#   bash scripts/reproduce.sh figures    figures and tables into $R/runs/figs*
#
# The runs are independent: one GPU (>= 40 GB for nside 256) or 32 CPU cores each.
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
set -e
cd $R/src
mkdir -p $R/runs
gpu() {
    if [ -n "$GPU_OPTS" ]; then sbatch $GPU_OPTS -J "$1" -o $R/runs/slurm.%j.log scripts/train_gpu.slurm "$@"
    else bash scripts/train_gpu.slurm "$@"; fi
}
cpu() {
    if [ -n "$CPU_OPTS" ]; then sbatch $CPU_OPTS -J "$1" -o $R/runs/slurm.%j.log scripts/train_cpu.slurm "$@"
    else bash scripts/train_cpu.slurm "$@"; fi
}
common="--steps 3000 --lr 1e-3 --seed 0 --realizations 4 --train-field Q --test-field U --log-every 50 --eval-every 500"
case "$1" in
runs)
    # Sect. 3, Table 1: nside 64, white noise, auto loss
    gpu sph_base  --nside 64 --noise uniform --loss data=1 $common
    gpu sph_cosmo --nside 64 --noise uniform --loss data=1 --gauge cosmo $common
    gpu sph_pr3   --nside 64 --noise pr3white --loss data=1 $common
    gpu sph_pr3x4 --nside 64 --noise pr3white --noise-scale 4 --loss data=1 $common
    gpu n64_base  --nside 64 --noise uniform --loss data=1 --conv faces $common
    # Sect. 4, Table 2: nside 256, FFP10 draws (train mc 10-49, test mc 0-9)
    gpu n256_data    --nside 256 --noise pr3sims --loss data=1 $common
    gpu n256_delouis --nside 256 --noise pr3sims --loss l1=1,l2=1,l3=1 $common
    cpu p256_l123 --nside 256 --j 4 --radius 10 --core 4 --patches 16 --centre-nside 8 --geometry-cache 800 \
        --levels 3 --channels 32,64,96 --loss l1=1,l2=1,l3=1 --realizations 4 --steps 3000 --lr 1e-3 --seed 0 \
        --log-every 20 --eval-every 500
    # Sect. 5: the PR3 maps themselves
    gpu r256_dl123 --nside 256 --real --loss data=1,l1=1,l2=1,l3=1 $common
    # the reference: the same losses minimised over the pixels of u with the L-BFGS of STL-Dev-MPI
    gpu l256_dl123 --nside 256 --real --loss data=1,l1=1,l2=1,l3=1 --realizations 4 --optimizer lbfgs \
        --lbfgs-rounds 10 --lbfgs-iter 50 --seed 0 --log-every 50 --eval-every 500
    # frozen noise biases (Delouis et al. 2022), then the spectrum loss and the 857 GHz conditioning
    gpu l256_fz --nside 256 --real --loss data=1,l1=1,l2=1,l3=1 --realizations 4 --optimizer lbfgs --bias frozen --bias-realizations 40 --bias-every 50         --lbfgs-rounds 10 --lbfgs-iter 50 --seed 0
    gpu r256_fz --nside 256 --real --loss data=1,l1=1,l2=1,l3=1 --bias frozen --bias-realizations 40 --bias-every 50 $common
    gpu r256_fz_cl857 --nside 256 --real --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45 --cond log857 --bias frozen --bias-realizations 40 --bias-every 50 $common
    gpu l256_fz_cl --nside 256 --real --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45 --realizations 4         --optimizer lbfgs --bias frozen --bias-realizations 40 --bias-every 50 --lbfgs-rounds 10 --lbfgs-iter 50 --seed 0
    ;;
figures)
    $PY scripts/make_figures.py $R/runs $R/runs/figs64
    $PY scripts/make_figures256.py $R/runs $R/runs/figs256
    $PY scripts/make_figures_real.py $R/runs/r256_dl123 $R/data/maps $R/runs/figs_real
    $PY scripts/make_figures_real.py $R/runs/l256_dl123 $R/data/maps $R/runs/figs_lbfgs
    for r in l256_fz r256_fz r256_fz_cl857 l256_fz_cl; do
        $PY scripts/make_figures_real.py $R/runs/$r $R/data/maps $R/runs/figs_$r
    done
    ls $R/runs/figs64 $R/runs/figs256 $R/runs/figs_real
    ;;
*)
    echo "usage: bash scripts/reproduce.sh runs | figures"; exit 1
    ;;
esac
