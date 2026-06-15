# Contributing to Chronos-ESM

Thanks for your interest in Chronos-ESM! It is an early **research preview**, so
expect rapid change and some rough edges. Contributions — bug reports, fixes,
documentation, validation, and new physics — are very welcome.

By contributing you agree that your contributions are licensed under the project's
[Apache License 2.0](LICENSE).

## Development setup

```bash
git clone https://github.com/bijanf/ESMB.git && cd ESMB
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest black isort flake8 mypy   # dev tools
```

The model runs on CPU out of the box (JAX). For GPU, install the matching
`jax[cuda]` wheel for your CUDA version (see the [JAX install guide](https://docs.jax.dev/en/latest/installation.html)).

## Running the tests

```bash
pytest -q
```

Please make sure the suite passes before opening a pull request.

## Code style

The repository uses `black`, `isort`, `flake8`, and `mypy` (configured in
`pyproject.toml` / `.flake8`). Run them before committing:

```bash
black .
isort --profile black .
flake8 .
mypy .
```

A `.pre-commit-config.yaml` is provided; `pre-commit install` will run these
automatically on each commit. CI runs the same checks on every push and pull request.

## Pull requests

1. Fork the repo and create a feature branch off `main`.
2. Make focused changes with clear commit messages.
3. Add or update tests for behavioural changes.
4. Ensure `pytest`, `black`, `isort`, `flake8`, and `mypy` all pass.
5. Update `CHANGELOG.md` and any affected docs.
6. Open a PR describing the change and its motivation.

## Reporting issues

Open a GitHub issue with: what you expected, what happened, a minimal way to
reproduce it (script + command), and your environment (OS, Python, JAX version,
CPU/GPU). For scientific/validation issues, include the relevant figure or scorecard.

## Scope notes

Some limitations are known and documented (see `README.md` and `CLAUDE.md`) — e.g.
the single-level atmosphere's missing synoptic systems, the experimental multi-level
`dinosaur` atmosphere, and the AMOC realism limitation (ocean mass-conservation).
Contributions toward these are especially welcome; please coordinate via an issue
first for large structural changes.
