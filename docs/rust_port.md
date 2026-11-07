# Plan: the whole training in Rust, on E and B

## Where the time goes

Profile of one step of the `r1k_eb` configuration at nside 128 on a T4 (8 steps, `cProfile`,
`CUDA_LAUNCH_BLOCKING=1`; nside 256 does not fit on 15 GB with PyTorch):

| Part | Time per step |
|---|---|
| scattering statistics, forward (d·u, u·u, T·u on E and B) | about 3 s |
| backward, mostly the adjoint of the statistics | about 5.9 s |
| spin-2 harmonic transforms (E/B of u, spectra) | about 0.9 s |
| network | negligible |
| once per run: engines, transform tables, imports / first bias estimate | about 100 s / 25 s |

The statistics are about 90 % of a step. The network is not worth optimising; the statistics and
the PyTorch / engine round trips are.

## What the native trainer already has

`stl_port.native.NativeUNet` (STL-Dev-MPI `main`, `docs/user-guide/native-unet.md`): the network,
its reverse pass, Adam and L-BFGS on one GPU; the terms `data`, `l1`, `l2`, `l3` with frozen biases
(zero on the unbiased coefficients) or live biases; statistics masks; conditioning `log857+sigma`;
the `cl` term through `SpectrumEngine`; one field per step (Q or U, alternately). At nside 256 on an
A100, 1.16 s per step against 3.40 s for PyTorch, 8.6 GB against 30.5 GB.

Missing for the best configuration (`r1k_eb`, `r256_joint_eebb`): the joint (Q, U) network, the
statistics on the E and B maps, and the EE/BB spectra. The last two need a spin-2 transform and its
adjoint in the training loop, which the GPU transform of the engine does not have.

## The spin-2 to spin-0 layers remove that need

Input layer `(Q, U) -> (E, B)`, output layer `(E, B) -> (Q, U)`: the network works on two scalar
maps, independent of the local frame (McEwen 2026, arXiv:2610.13892: spin-changing convolutions,
equivariant, one spin-weighted transform each).

- The residual stays in (Q, U): `u = d - T^-1 r(T d)`, `T` the E/B transform. `d` itself never goes
  through the transforms, so the band limit and the HEALPix quadrature only affect the correction.
- The loss is on `u_E = E_d - r_E`, `u_B = B_d - r_B`. `E_d`, `B_d`, the half missions and the noise
  simulations are transformed **once**, before training (`healpy.map2alm(..., pol=True, iter=3)`).
- Every term is then scalar: `data`, `l1`, `l2`, `l3` on E and B, `cl` as the scalar spectra of the
  E and B maps (EE, BB). No transform in the training loop, no adjoint to write.
- This is what the native trainer computes already, with E and B in place of Q and U.

## Steps

1. **Zero-code test (PyTorch, nside 64, garoupe).** Write E/B files under the Q/U names, train
   `train_denoise.py --joint --pol-stats qu` on them, bring u back to (Q, U), and compare with
   `t64_jeb1k` (statistics on E/B of u). Expected: the same spectra and noise removed, without the
   transforms in the step.
2. **Native, single field (nside 64, garoupe).** `train_native_unet.py` on the same E/B files: E and
   B alternate, as Q and U today. Compare with step 1, and time it.
3. **Native joint network.** `maps=2` in `NativeUNet`: two input and two output channels, one scale
   per map; a joint step whose loss is the sum over E and B, and whose cotangent is `[2, npix]`.
   The torch `UNet(maps=2)` state dict loads both ways (as today for `maps=1`).
4. **Debiasing, the same for every term.** One set of draws per refresh for all terms and both
   fields: `n_E`, `n_B` from the same simulation and sign, the half-mission pair from the same
   simulation; biases only on the coefficients with a noise bias (`unbiased()`); refreshed at the
   same step for E and B. `l2` (`Phi(d, u)` against `Phi(u + n, u)`) is kept; the terms on `d - u`
   (`res`, `rcl`, `xcl`) are dropped from the configurations.
5. **Driver.** `train_native_unet.py --pol eb`: transforms the inputs once, writes `eval_{Q,U}.npz`
   in (Q, U) after the output layer, and `model.pt` with the layer declared, so that
   `stlia.unet.load` and the figure scripts read it unchanged.
6. **Profile and optimise (A100, with agreement).** nside 256, then 512: number of statistics per
   step, engine cache size, single-precision statistics where the χ² allows, bias refresh period.
