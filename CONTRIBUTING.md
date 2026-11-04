# Contributing

## The documentation follows the code

The documentation is part of every change, not a later task. A change is complete only when all of the following hold:

1. **The docstrings are current.** The API pages are generated from them.
2. **The command-line help is current.** Every option of `stlia.train_denoise.parser` and `stlia.train_patches.parser` has a `help=` string, and the command-line pages are generated from these parsers.
3. **The pages that describe the behaviour are updated** in `docs/`:
   - the method, for a loss, the network or the patches;
   - the user guide, for options, data, outputs or resources;
   - `REPRODUCE.md`, for anything a rerun of the paper depends on (commands, versions, expected values).
4. **`CHANGELOG.md` has an entry** under *Unreleased*, saying what changed and which pages were updated.
5. **The documentation builds without warnings**:

   ```bash
   sphinx-build -W --keep-going -b html docs docs/_build
   ```

6. **The tests pass**:

   ```bash
   pytest
   ```

   Behaviour that can be checked without a GPU gets a test in `tests/`.

The CI enforces points 4 to 6 on every push and pull request:

- the `docs` job builds the site with warnings as errors, and publishes it on GitHub Pages from `main`;
- the `docs-updated` job fails when files under `stlia/`, the training wrappers, `scripts/` or `pyproject.toml` change without any change under `docs/`, in `REPRODUCE.md` or in `CHANGELOG.md`.

For a change that really needs no documentation (a pure refactor, a typo), put `[no-docs]` in the commit message or the pull-request title.

## Development setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e ".[test,docs]"
pytest
sphinx-build -W --keep-going -b html docs docs/_build
```

The tests that need the STL-Dev-MPI engine (marker `stl`) are skipped when `stl_port.native` cannot be imported; the installation page of the documentation explains how to build it.

## Style

- **Code.** Python 3.10+. Follow the existing style: short functions, NumPy-style docstrings, comments that say why.
- **Docs.** Markdown (MyST) for the pages, reStructuredText only where Sphinx directives need it.
