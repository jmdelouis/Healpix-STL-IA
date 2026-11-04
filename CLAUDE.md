# Instructions for coding agents

Read `CONTRIBUTING.md` first. The rules that matter most:

- **Docs with the code.** Every code change updates, in the same commit:
  - the docstrings;
  - the `help=` strings of the command-line parsers;
  - the pages of `docs/` that describe the changed behaviour;
  - `REPRODUCE.md`, if a rerun of the paper depends on it;
  - an entry in `CHANGELOG.md` under *Unreleased*.
- **Check before pushing.** Run `pytest` and `sphinx-build -W --keep-going -b html docs docs/_build`. The CI fails on warnings and on code changes without a docs or changelog change.
- **The paper.** `text/` is an Overleaf project synchronised by `git subtree`: push it with `git subtree split --prefix=text` after checking that Overleaf has no new commits, and never `subtree pull`.
- **Heavy work.** Data and training live on the user's HPC work directory, not in this repository: see `REPRODUCE.md`.
