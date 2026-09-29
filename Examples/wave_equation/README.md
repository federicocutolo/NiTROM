# Wave-equation NiTROM example

Structure-preserving reduced-order modeling of the 1D linear wave equation, cast
as a **canonical Hamiltonian system**. Two baselines (VC-H-OpInf, Petrov–Galerkin)
warm-start and benchmark a **NiTROM** (Nonlinear Trajectory-based ROM) trained by
Riemannian optimization of the symplectic basis `Φ` and the reduced Hamiltonian
tensor `A2`.

## The model

State `x = [q; p]` (displacement, momentum). The FOM is linear and canonical:

```
ẋ = J W x,        H = ½ xᵀ W x,        J = [[0, I], [-I, 0]],   W = blkdiag(-K, I)
```

`K` is the spatial Laplacian stencil. Because the FOM is linear, the exact flow
over a uniform step is `expm(dt·JW)` — exact and symplectic (energy conserved to
machine precision), so we never need an implicit solver for data generation.

Two test cases share all code (`wave_cases.py` picks one):
- **Dirichlet** (`u(0)=u(L)=0`): single *traveling* Gaussian pulse (directional
  momentum), `N=128`, `L=2`, `c=1`.
- **Periodic** (Gruber–Tezaur): *zero-momentum* Gaussian bump that splits into
  **two counter-propagating waves**; minimal-image (wrap-around) IC keeps it
  smooth across the seam. `L=1`, `c=0.1`.

## Pipeline / run order

```
python Dirichlet/main.py     # (or Periodic/main.py) -> generate data + basis + plots
mpiexec -n N python -u train_nitrom.py   # train NiTROM (N <= n_traj)
python test_nitrom.py        # FOM vs ROMs: q(L/2,t), error curves, energy drift, GIFs
```

| File | Role |
|---|---|
| `wave_foms.py` | FOM operators (`W`, `J`), exact `expm` integrator, Hamiltonian. |
| `wave_cases.py` | Registry: per-case physics, I/O folders, IC **sampling box**. |
| `wave_sampling.py` | The IC sampler — *how* Gaussian-pulse ICs are drawn. |
| `Dirichlet/main.py`, `Periodic/main.py` | Generate trajectories, velocities, cotangent-lift basis. |
| `wave_roms.py` | Baselines: VC-H-OpInf and Petrov–Galerkin. |
| `train_nitrom.py` | Progressive-window Riemannian CG training of `(Φ, A2)`. |
| `test_nitrom.py` | Evaluation + animations on training and held-out test sets. |
| `rank_sweep.py`, `nitrom_pipeline.py` | Rank study / reusable training driver. |

## The ROMs

- **Cotangent-lift basis** (Peng–Mohseni): POD of the stacked `[Q | P]` block, so
  the reduced symplectic form `Ĵ = ΦᵀJΦ` is canonical (`cond = 1`) by construction.
- **VC-H-OpInf** (Gruber & Tezaur): symmetric reduced Hamiltonian `A_bar` from the
  Lyapunov system `G A_bar + A_bar G = M + Mᵀ`, using **exact** velocity snapshots
  `Ẋ = JWX` (not finite differences). `H` is quadratic, so only `A_bar` is inferred.
- **Petrov–Galerkin**: intrusive symplectic projection of `W`.
- Each baseline is exposed in two roles — `*_init_A2` (NiTROM warm-start) and
  `*_dynamics` (`ż = A z` for integration/plots). `A_bar` is a cheap Lyapunov solve,
  so it is recomputed on demand rather than persisted (removes a train↔test ordering
  dependency).

## Design decisions & gotchas

- **Single source of truth.** `wave_cases` owns the IC ranges; `wave_sampling` owns
  the draw. Training (`main.py`) and the held-out test set (`test_nitrom.py`) draw
  from the *same* box, differing only by `seed`.
- **Progressive training window.** Cost is `mean reconstruction error per snapshot`.
  Build the `optimization_objects` **inside** each phase so the weights scale by the
  *current* window's snapshot count — otherwise the cost grows with the window.
  Long-horizon adjoints vanish/explode, so short→long windows is the stable recipe.
- **MPI is over trajectories.** Ranks ≤ `n_traj`. `cost()` runs an internal
  `allgather`, so **every** rank must call it — evaluating on rank 0 only
  desynchronizes the collectives.
- **`dt_rom` tracks the reduced fastest frequency, not the mesh.** The ROM is an
  `r`-dim ODE with no grid; size `dt_rom ≈ 2π/(K·ω_max)`, `ω_max = max|Im λ(A_dyn)|`
  (printed by the stability check), `K ≈ 20–50`. Refining `N` does **not** tighten
  `dt_rom` unless you also enrich the basis with higher-frequency modes.
- **SVD spectrum ↔ reducibility.** A *steep* `1−Σσ²` decay ⇒ low-rank/reducible
  (small `var_rank`); a *flat* decay ⇒ transport-dominated (slow Kolmogorov n-width)
  — driven mainly by how far the pulse travels (`c·T/L`). Note the σ floor is tied to
  `dx` (`sigma_dx_factor·dx`), so changing `N` also changes the IC family; decouple it
  for clean resolution studies.
- **For a linear, well-resolved problem NiTROM ≈ VC-H-OpInf.** OpInf already recovers
  the (near-)exact reduced operator and the error is projection-dominated, so the
  optimizer barely moves. NiTROM earns its keep with nonlinearity (`poly_comp=[1,2]`),
  more aggressive reduction (small `r`), or longer/harder horizons.
- **Energy-drift plots use each model's OWN invariant** in latent space
  (OpInf: `½ zᵀA_bar z`, NiTROM: `zᵀA2_nit z`); full-space drift uses `½ xᵀW x`.
- **Videos play in REAL TIME.** `fps = n_frames / T_SIM`, so wall-clock == physical
  time and the title clock advances 1 s ↔ 1 time unit. `T_SIM = max(8, L/c)` ensures
  the GIF covers at least one full wave period instead of cutting off mid-traversal.
