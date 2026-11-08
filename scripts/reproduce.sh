#!/bin/bash
# Every run and figure of the paper (REPRODUCE.md).  After the environment
# (scripts/env.sh or a site file), the build and the data:
#
#   bash scripts/reproduce.sh runs       submit every run (Slurm), or run them
#                                        one after the other when GPU_OPTS is empty
#   bash scripts/reproduce.sh figures    figures and tables into $RUNS/figs*
#
# The runs are independent: one GPU (>= 40 GB for nside 256) or 32 CPU cores each.
: "${R:?source scripts/env.sh (or a site file) first}"
source $R/src/scripts/env.sh
set -e
cd $R/src
mkdir -p $RUNS
gpu() {
    if [ -n "$GPU_OPTS" ]; then sbatch $GPU_OPTS -J "$1" -o $RUNS/slurm.%j.log scripts/train_gpu.slurm "$@"
    else bash scripts/train_gpu.slurm "$@"; fi
}
gpup() {
    if [ -n "$GPU_OPTS" ]; then sbatch $GPU_OPTS -J "$1" -o $RUNS/slurm.%j.log scripts/train_patches_gpu.slurm "$@"
    else bash scripts/train_patches_gpu.slurm "$@"; fi
}
cpu() {
    if [ -n "$CPU_OPTS" ]; then sbatch $CPU_OPTS -J "$1" -o $RUNS/slurm.%j.log scripts/train_cpu.slurm "$@"
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
    gpu r256_fz --nside 256 --real --loss data=1,l1=1,l2=1,l3=1 --bias frozen --bias-realizations 40 --bias-every 50 $common
    STL=stl_sp2 gpu r256_fz_cl857 --nside 256 --real --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45,60 --cond log857 --bias frozen --bias-realizations 40 --bias-every 50 $common
    STL=stl_sp2 gpu l256_fz_cl --nside 256 --real --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45,60 --realizations 4         --optimizer lbfgs --bias frozen --bias-realizations 40 --bias-every 50 --lbfgs-rounds 10 --lbfgs-iter 50 --seed 0
    # live biases with the spectrum loss: the cross spectrum is unbiased in either scheme
    gpu r256_cl857 --nside 256 --real --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45 --cond log857 $common
    gpu l256_cl --nside 256 --real --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45 --realizations 4         --optimizer lbfgs --lbfgs-rounds 10 --lbfgs-iter 50 --seed 0
    # one network for (Q, U), with the EE/BB spectrum loss (STL-Dev-MPI >= 92f2aef, built in $R/stl_sp2)
    STL=stl_sp2 gpu r256_joint_eebb --nside 256 --real --joint --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-spin 2         --cl-backend stl --cl-masks 0,30,45,60 --cond log857 $common
    # the native U-Net of STL-Dev-MPI (Rust, commit b580210, built in $R/stl_nu), as r256_fz
    if [ -n "$GPU_OPTS" ]; then
        STL=stl_nu sbatch $GPU_OPTS -J r256_native -o $RUNS/slurm.%j.log scripts/train_native.slurm r256_native             --nside 256 --optimizer adam --steps 3000 --lr 1e-3 --bias-every 50 --bias-realizations 40             --loss data=1,l1=1,l2=1,l3=1 --sims 10-49 --cond none --eval-every 500 --engine-cache-bytes 0
    fi
    # the joint native network (STL-Dev-MPI main 45d3e17, built in $R/stl_nj) with the options of
    # r1k_eb over 3000 steps: statistics on chi = conj-eth^2 (Q + iU) (--pol spin), or a network on the E and B maps (--pol eb)
    for pol in spin eb; do
        STL=stl_nj sbatch $GPU_OPTS -J n256_$pol -o $RUNS/slurm.%j.log scripts/train_native.slurm n256_$pol             --nside 256 --optimizer adam --steps 3000 --lr 1e-3 --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45             --cl-masks 0,30,45,60 --cond log857+sigma --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-49             --eval-every 500 --log-every 50 --engine-cache-bytes 0 --seed 0 --input-scale map --pol $pol
    done
    # the same on the E and B maps with 16-64 channels per level, as r3k_eb_c16
    STL=stl_nj sbatch $GPU_OPTS -J n256_eb_c16 -o $RUNS/slurm.%j.log scripts/train_native.slurm n256_eb_c16 \
        --nside 256 --optimizer adam --steps 3000 --lr 1e-3 --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45 \
        --cl-masks 0,30,45,60 --cond log857+sigma --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-49 \
        --eval-every 500 --log-every 50 --engine-cache-bytes 0 --seed 0 --input-scale map --pol eb --channels 16,32,48,64,64,64
    # the same network on (Q, U), statistics on E and B of the output (STL-Dev-MPI commit 596e3e9, GPU E/B front end)
    STL=stl_ebqu sbatch $GPU_OPTS -J n256_ebqu_c16 -o $RUNS/slurm.%j.log scripts/train_native.slurm n256_ebqu_c16 \
        --nside 256 --optimizer adam --steps 3000 --lr 1e-3 --loss data=1,l1=1,l2=1,l3=1,cl=1 --stat-masks 0,20,30,45 \
        --cl-masks 0,30,45,60 --cond log857+sigma --bias-warmup 200 --bias-realizations 40 --bias-every 50 --sims 10-49 \
        --eval-every 500 --log-every 50 --engine-cache-bytes 0 --seed 0 --input-scale map --pol ebqu --channels 16,32,48,64,64,64
    # spread between trainings: r256_native again with seeds 1-3 (seed 0 is r256_native)
    if [ -n "$GPU_OPTS" ]; then
        for s in 1 2 3; do
            STL=stl_nu sbatch $GPU_OPTS -J r256_native_s$s -o $RUNS/slurm.%j.log scripts/train_native.slurm r256_native_s$s                 --nside 256 --optimizer adam --steps 3000 --lr 1e-3 --bias-every 50 --bias-realizations 40                 --loss data=1,l1=1,l2=1,l3=1 --sims 10-49 --cond none --eval-every 500 --engine-cache-bytes 0 --seed $s
        done
    fi
    # masked statistics (STL-Dev-MPI >= 9038114, built in $R/stl_mk: STL=stl_mk)
    STL=stl_mk gpu r256_fz_cl857_mk --nside 256 --real --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45         --stat-masks 0,20,30,45 --cond log857 --bias frozen --bias-realizations 40 --bias-every 50 $common
    # the best configuration (r256_fz_cl857_mk) with the local noise level as a second conditioning map and
    # 500 steps of live biases first; then the same as one (Q, U) network with every term on the E and B maps
    STL=stl_sp2 gpu r256_sig_wu --nside 256 --real --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45,60         --stat-masks 0,20,30,45 --cond log857+sigma --bias frozen --bias-warmup 500 --bias-realizations 40 --bias-every 50 $common
    STL=stl_sp2 gpu r256_eb --nside 256 --real --joint --pol-stats eb --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-masks 0,30,45,60         --stat-masks 0,20,30,45 --cond log857+sigma --bias frozen --bias-warmup 500 --bias-realizations 40 --bias-every 50 $common
    # the same with the local spin-2 statistics chi = conj-eth^2 (Q + iU) of the engine (commit f4cf030)
    STL=stl_spin gpu r256_chi --nside 256 --real --joint --pol-stats chi --loss data=1,l1=1,l2=1,l3=1,cl=1 --cl-spin 2         --cl-backend stl --cl-masks 0,30,45,60 --stat-masks 0,20,30,45 --cond log857+sigma --bias frozen --bias-warmup 500         --bias-realizations 40 --bias-every 50 $common
    # 1000 steps: the best configuration, then L2 replaced by Phi(d - u) = Phi(n) (res) with the spectra
    # of the 857 GHz map against u (tcl) and of d - u against the noise of d (rcl)
    short="--steps 1000 --lr 1e-3 --seed 0 --realizations 4 --train-field Q --test-field U --log-every 50 --eval-every 250"
    best="--nside 256 --real --cl-masks 0,30,45,60 --stat-masks 0,20,30,45 --cond log857+sigma --bias frozen         --bias-warmup 200 --bias-realizations 40 --bias-every 50"
    STL=stl_sp2 gpu r1k_ref $best --loss data=1,l1=1,l2=1,l3=1,cl=1 $short
    STL=stl_sp2 gpu r1k_res $best --loss data=1,l1=1,l3=1,res=1,cl=1,tcl=1,rcl=1 $short
    # the best configuration as one (Q, U) network with every term, the spectrum term included, on the E and B maps
    STL=stl_sp2 gpu r1k_eb $best --joint --pol-stats eb --loss data=1,l1=1,l2=1,l3=1,cl=1 $short
    STL=stl_sp2 gpu r1k_eb_s1 $best --joint --pol-stats eb --loss data=1,l1=1,l2=1,l3=1,cl=1 ${short/--seed 0/--seed 1}
    # the per-map L-BFGS reference of r1k_eb, and r1k_eb with a narrow network
    STL=stl_sp2 gpu l1k_eb $best --joint --pol-stats eb --loss data=1,l1=1,l2=1,l3=1,cl=1 $short --optimizer lbfgs         --lbfgs-rounds 10 --lbfgs-iter 50 --bias-warmup 0
    STL=stl_sp2 gpu r1k_eb_c16 $best --joint --pol-stats eb --loss data=1,l1=1,l2=1,l3=1,cl=1 $short --channels 16,32,48,64,64,64
    # the narrow E/B network over 3000 steps, to compare with r256_joint_eebb at equal training
    STL=stl_sp2 gpu r3k_eb_c16 $best --joint --pol-stats eb --loss data=1,l1=1,l2=1,l3=1,cl=1         ${short/--steps 1000/--steps 3000} --channels 16,32,48,64,64,64
    # the same with u and d - u decorrelated (xcl)
    STL=stl_sp2 gpu r1k_ebx $best --joint --pol-stats eb --loss data=1,l1=1,l2=1,l3=1,cl=1,xcl=1 $short
    # the loss of r1k_eb on the 48 discs of the nside 2 cells, and the full-sky control with the same network (nside 64)
    pt="--nside 64 --real --joint --pol-stats eb --cond log857 --bias frozen --bias-realizations 8 --bias-every 50         --loss data=1,l1=1,l2=1,l3=1,cl=1 --levels 3 --channels 32,64,96 --steps 1000 --lr 1e-3 --log-every 50 --sims-train 10-15"
    STL=stl_sp2 gpup pt64_cells $pt --eval-every 250 --centre-nside 2 --radius 27 --core 17 --j 4 --patches 8
    STL=stl_sp2 gpu t64_jebl3p $pt --eval-every 500 --noise pr3sims --sims-test 10-15 --realizations 4 --cl-masks 0,30,45,60         --train-field Q --test-field U --norm pixel
    STL=stl_mk gpu l256_fz_mk --nside 256 --real --loss data=1,l1=1,l2=1,l3=1 --stat-masks 0,20,30,45         --realizations 4 --optimizer lbfgs --bias frozen --bias-realizations 40 --lbfgs-rounds 10 --lbfgs-iter 50 --seed 0
    ;;
figures)
    # figures of the paper: the reference run, with the GNILC dust map and the BICEP2/Keck field
    # (GNILC from the Planck Legacy Archive or IRSA, bk14_mask_gal_n0512.fits from bicepkeck.org, in $R/data/ext)
    E=$R/data/ext
    $PY scripts/prepare_gnilc.py $E/COM_CompMap_IQU-thermaldust-gnilc-varres_2048_R3.00.fits $E
    $PY scripts/make_figures_real.py $RUNS/n256_ebqu_c16 $R/data/maps $RUNS/figs_ref --gnilc $E \
        --field-mask $E/bk14_mask_gal_n0512.fits
    # figures of the earlier runs (history of the tests)
    $PY scripts/make_figures.py $RUNS $RUNS/figs64
    $PY scripts/make_figures256.py $RUNS $RUNS/figs256
    $PY scripts/make_figures_real.py $RUNS/r256_dl123 $R/data/maps $RUNS/figs_real
    $PY scripts/make_figures_real.py $RUNS/l256_dl123 $R/data/maps $RUNS/figs_lbfgs
    for r in r256_fz r256_fz_cl857 l256_fz_cl r256_cl857 l256_cl r256_joint_eebb r256_native r256_fz_cl857_mk l256_fz_mk r256_sig_wu r256_eb r256_chi r1k_ref r1k_res r1k_eb r1k_eb_s1 r1k_ebx l1k_eb r1k_eb_c16 r3k_eb_c16; do
        $PY scripts/make_figures_real.py $RUNS/$r $R/data/maps $RUNS/figs_$r
    done
    $PY scripts/make_figure_highlat.py $R/data/maps $RUNS/highlat_spectra.pdf --lmax 449 --bins 10 \
        "$RUNS/r256_fz_cl857:network, Q and U" "$RUNS/r256_joint_eebb:network (Q,U), EE/BB" "$RUNS/l256_fz_cl:L-BFGS, Q and U"
    ls $RUNS/figs64 $RUNS/figs256 $RUNS/figs_real
    ;;
*)
    echo "usage: bash scripts/reproduce.sh runs | figures"; exit 1
    ;;
esac
