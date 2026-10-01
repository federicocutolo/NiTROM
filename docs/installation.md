# Installation

```bash
git clone https://github.com/albertopadovan/NiTROM.git
cd NiTROM
pip install -e .              # runtime deps: numpy, scipy, numba, matplotlib, torch, dill
pip install -e ".[dev]"       # + pytest, ruff
pip install -e ".[docs]"      # + sphinx toolchain, to build this documentation
```

Requires Python ≥ 3.10. `mpi4py` is optional and only needed for MPI-parallel training
on the NumPy backend; `torch` is imported lazily, so a NumPy-only session never pulls
it in.

## Building the documentation

The API pages embed inheritance diagrams rendered with
[Graphviz](https://graphviz.org/), so the `dot` binary must be on your `PATH`
(`brew install graphviz`, `conda install graphviz`, or `apt-get install graphviz`).
Then:

```bash
cd docs
make html        # output in docs/_build/html
```

## Running the tests

```bash
pytest                        # full suite
pytest tests/optimization -q  # just the optimization tests
```

The suite covers the backend shim, training-data sharding, projections, latent-space
models, time steppers and their adjoints, manifold geometry, the Riemannian L-BFGS, and
finite-difference gradient checks for every inference module. Linting uses `ruff` with
the configuration in `pyproject.toml`.
