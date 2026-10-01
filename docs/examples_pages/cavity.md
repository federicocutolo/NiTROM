# Lid-driven cavity

2D lid-driven cavity flow at $Re = 8300$ on a $100\times100$ grid, reduced to
$r = 50$ (`examples/cavity/`) and $r = 30$ (`examples/cavity_30/`). The training
scripts compare Galerkin projection, OpInf, GAS-OpInf, NiTROM, and GAS-NiTROM side by
side and save self-contained ROM checkpoints (bases + operators) as pickles.

```{admonition} Requirements
:class: note
The full-order model is Numba-accelerated (`pip install numba`), and the trajectory
data is **not** committed — run `compute_baseflow.py` and `generate_data.py` first.
Training at $r=50$ is a long-running job intended for a workstation or cluster
(`mpiexec` / `torchrun` parallelism over trajectories).
```

## Workflow

1. `compute_baseflow.py` — steady base flow for the chosen Reynolds number;
2. `generate_data.py` — integrate and store training trajectories
   (`classes_cavity.py`, `numba_operators.py`, and `linear_operators.py` implement
   the FOM);
3. `train_opinf.py`, `train_nitrom.py`, `train_gasnitrom.py` — the same training
   ladder as the toy model, at scale (`train_opinf_sweep.py` sweeps the
   regularization);
4. `read_results.py`, `post_process.py`, `plot_baseflow.py` — evaluation and figures.

## Training script

The NiTROM training driver mirrors the executed toy-model pipeline — the only
differences are the FOM callables and the problem sizes:

```{literalinclude} ../../examples/cavity/train_nitrom.py
:language: python
:caption: examples/cavity/train_nitrom.py
```
