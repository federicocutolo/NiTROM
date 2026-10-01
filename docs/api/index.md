# API reference

The package is organized in layers: data containers feed a *(projection, model)* pair,
an inference module turns the pair into a scalar cost with analytic gradients, and the
training loop or the closed-form OpInf solver optimizes it. Backends and time steppers
are cross-cutting services used by every layer.

| Layer | Subpackage / module | Canonical imports |
| --- | --- | --- |
| Training data | `nitrom.training_data` | `TrainingPool`, `TrainingData` |
| Projections | `nitrom.projections` | `LinearProjection`, `VarconLinearProjection`, `PolynomialProjection` |
| Latent-space models | `nitrom.latent_space_models` | `PolynomialModel`, `GasPolynomialModel`, `AtrPolynomialModel`, `HamiltonianPolynomialModel` |
| Parameter registry | `nitrom.roms` | `ParamRegistry` |
| Optimization | `nitrom.optimization` | `NitromModule`, `OpInfModule`, `PolyManifoldInfModule`, `train`, `solve_opinf`, `perform_POD` |
| Time steppers | `nitrom.time_steppers` | `solve_ivp`, `solve_adjoint_ivp_discrete`, `linear_symplectic_solve` |
| Backend & support | `nitrom.backend`, `nitrom.utils` | `set_backend`, `setup_distributed`, `compute_POD` |

All of the names above are re-exported at the top level, so `from nitrom import train`
works too.

```{toctree}
:maxdepth: 2

data
projections
latent_space_models
roms
optimization
time_steppers
support
```
