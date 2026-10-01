# Data

Two classes separate *owning* trajectory data from *using* it.

## TrainingPool: load and shard

{class}`~nitrom.training_data.TrainingPool` loads trajectory snapshots, optional
per-trajectory weights, steady forcing callables, and time derivatives from disk, and
shards the trajectories across ranks — auto-detecting `MPI.COMM_WORLD` (NumPy backend)
or the `torch.distributed` process group (PyTorch backend). Each rank ends up with its
contiguous slice of trajectories (`pool.my_n_traj` of `n_traj`), and all collective
reductions later in training are handled by the backend layer.

### On-disk conventions

File locations are given as `%`-format strings indexed by trajectory number:

| Constructor argument | Typical pattern | Contents |
| --- | --- | --- |
| `fname_traj` | `traj_%03d.npy` | snapshots, one array per trajectory |
| `fname_time` | `time.npy` | the shared, uniform time grid |
| `fname_weights` | `weight_%03d.npy` | optional per-trajectory weights |
| `fname_forcing` | `forcing_%03d.pkl` | optional dill-pickled forcing callables |
| `fname_derivs` | `deriv_%03d.npy` | optional time derivatives |

If derivative files are missing, derivatives are computed with high-order finite
differences (`nitrom.finite_difference`) and **cached back to disk** (at the
`fname_derivs` path, or at a `deriv_<idx>.npy` path derived from `fname_traj` when it
was omitted), so later runs load instead of recomputing. The full parameter-by-parameter
contract lives in the {class}`~nitrom.training_data.TrainingPool` docstring.

## TrainingData: the optimizer's view

{class}`~nitrom.training_data.TrainingData` wraps a pool without copying it and decides
what one *training run* sees:

- `which_trajs` — the subset of (global) trajectory indices to train on;
- `percent_time_length` — truncate every trajectory to a fraction of its horizon;
- `leggauss_deg` — Gauss–Legendre quadrature order for the adjoint time integrals;
- `nsave_rom` — how many latent snapshots the time-marched ROM saves between data
  points;
- virtual trajectories — each physical trajectory can be expanded into several
  time-shifted copies, cheaply enlarging the training set.

Several `TrainingData` views (e.g. short-horizon for warm-up, full-horizon for
fine-tuning) can share one loaded pool.
