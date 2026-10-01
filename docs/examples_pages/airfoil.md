# Airfoil

Flow over an airfoil reduced to $r = 50$ (`examples/airfoil/`), including the
attracting-trapping-region model
({class}`~nitrom.latent_space_models.AtrPolynomialModel`) among the trained variants,
plus GPU timing utilities (`time_calls_gpu.py`) and movie generation for the
full-order flow, the forcing, and the ROM predictions.

```{admonition} Requirements
:class: note
This case ships **no data generator** — the training trajectories come from an
external CFD solver and are not committed. The scripts document the expected on-disk
layout (the standard `traj_%03d.npy` / `time.npy` convention of
{class}`~nitrom.training_data.TrainingPool`).
```

## Workflow

1. `train_opinf.py` → `train_nitrom.py` / `train_gasnitrom.py` — the training ladder
   (`train_opinf_sweep.py` sweeps the regularization);
2. `read_results.py` — error metrics across the trained ROMs;
3. `make_fom_movie.py`, `make_forcing_movie.py`, `make_prediction_movies.py`,
   `plot_baseflow.py` — visualization.

## Training script

```{literalinclude} ../../examples/airfoil/train_nitrom.py
:language: python
:caption: examples/airfoil/train_nitrom.py
```
