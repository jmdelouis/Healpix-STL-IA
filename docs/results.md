# Results: the paper

The experiments are reported in the paper, *Amortised scattering denoising on the sphere*. Its LaTeX source and figures
are in [`text/`](https://github.com/jmdelouis/Healpix-STL-IA/tree/main/text), built with `latexmk -pdf main.tex`. The
documentation gives the options; the paper gives what they produced.

## The runs

| run | what it is | paper |
|---|---|---|
| `n256_ebqu_c16_cl3n` | the reference: PR3 353 GHz $Q$, $U$ at $\nside=256$ | Sects. 3 and 5, Table 3, Figs. 3-6 |
| `sim_d12_ebqu_cl3n` | the same options on a known sky (PySM d12 + FFP10 noise), applied to 9 unseen noises | Sect. 4, Table 2, Figs. 1-2 |
| `n512_ebqu_c16_cl3n_g` | the reference at $\nside=512$, with the spike guard | Sect. 5, Table 3, Appendix B |
| `n256_ebqu_c16`, `n256_ebqu_c16cl`, `sim_d12_ebqu`, `n512_ebqu_c16` | the same with the spectrum term up to $2\nside-1$ | Tables 2-3 |
| the runs of Sect. 7 | every alternative tested, in order | Sect. 7, Table 5 |

{doc}`reproduce` rebuilds each of them. {doc}`options/native-trainer` gives, for each option, the experiment of the
paper that varied it.

## In short

- **Known sky.** Over $\vert b\vert>30^\circ$, the spectra of the truth are recovered to 4 % up to $\ell=300$.
  - The pixel error is 0.32 times the noise at $\vert b\vert>60^\circ$, the same on nine noise realisations not seen in
    training.
  - Where the noise dominates, $u$ has the statistics of the sky but is only partly correlated with it.
- **PR3 maps, $\nside=256$.** Over $\vert b\vert>30^\circ$, the EE and BB spectra of $u$ follow the half-mission cross
  spectrum to 2 % up to $\ell=300$ and to 6 % for $300\le\ell<450$.
  - The removed part has 0.94-1.03 times the noise rms in every latitude band above $30^\circ$.
  - The per-map L-BFGS solution of the same loss gives the same spectra but removes less noise on $U$.
- **Cost.** A training takes 2.2 h on one A100, 3.5 times less than with a PyTorch network calling the same engine.
- **Spread.** Two trainings with the same loss end on maps that differ by a large fraction of the noise.

The open problems are listed in `HANDOFF.md` and in "What to try first" of {doc}`options/index`:
- the BB excess of the faint sky;
- the rise of $u$ above $\ell=2\nside$ at 512;
- the large-scale residual at 512;
- the Adam spikes;
- nside 1024, not trained yet.
