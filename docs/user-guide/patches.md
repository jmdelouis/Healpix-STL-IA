# Training on disc patches

`stlia-train-patches` (or `python train_patches.py`) trains the network on random discs instead of whole maps. All options are listed in {doc}`../cli`; the principle is in {doc}`../method`.

```bash
# disc statistics on 32 CPU cores: about 3.3 s per step at nside 256
stlia-train-patches --data maps --out runs/p256 --nside 256 --j 4 --radius 10 --core 4 \
    --patches 16 --centre-nside 8 --levels 3 --channels 32,64,96 --device cpu --threads 32

# disc statistics on the GPU (STL-Dev-MPI >= 20329e1)
stlia-train-patches --data maps --out runs/p256g --nside 256 --j 4 --radius 10 --core 4 \
    --patches 16 --centre-nside 8 --levels 3 --channels 32,64,96 \
    --stl-backend gpu --device cuda:0 --cache-bytes 6e10
```

Every step works as follows:

1. It draws `--patches` centres among the pixel centres of `--centre-nside`, giving 768 candidates at `--centre-nside 8`. A finite set lets the engine reuse its cached disc geometries.
2. It draws one observation (a training simulation) and `--realizations` noise triples $(\tilde n, \tilde n_1, \tilde n_2)$ from other training simulations.
3. It runs the network on the discs and computes the three cross losses. All the pairs of the step go through one forward and one backward call of the engine.

The disc losses currently have only `l1`, `l2` and `l3`, without a mean or variance term. In the paper, the mean of $u$ drifts away from that of the sky during patch training.

## Memory and speed

- **CPU engine**: the geometry caches set the memory. `--geometry-cache 800` with `--centre-nside 8` keeps the resident memory near 45 GB at $N_{\rm side}=256$.
- **GPU engine**: each disc geometry costs about 90 MB at $N_{\rm side}=256$, $R=10^\circ$. Since STL-Dev-MPI 20329e1, the cache budget is fixed when the engine starts. On a GPU shared with the network, set it explicitly with `--cache-bytes`: about $6\times10^{10}$ on an 80 GB A100 (all 768 centres), $3\times10^{9}$ on a 16 GB T4. From 4ae7b9d on, the engine instead keeps `--reserve-bytes` free beyond its cache, measured when it adds a disc. In both cases, leave room for the full-sky evaluation of the network, which needs several GB at $N_{\rm side}=256$.

On a T4, with a 3 GB cache, a step takes 2.0 s; the CPU engine takes 3.3 s on 32 cores.
