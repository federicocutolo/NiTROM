# Getting started

The snippet below is the whole NiTROM workflow on a small problem: load trajectory
data, build an initial projection from POD, choose a latent-space model, and train the
bases and operators jointly.

```python
import numpy as np
from nitrom.backend import set_backend
from nitrom.training_data import TrainingPool, TrainingData
from nitrom.utils import compute_POD
from nitrom.projections import LinearProjection
from nitrom.latent_space_models import PolynomialModel
from nitrom.roms import ParamRegistry
from nitrom.optimization import NitromModule, train

set_backend("numpy")          # or "torch" for autograd / GPU

# 1. Load trajectories from disk (sharded across ranks under mpiexec/torchrun)
pool = TrainingPool(
    n_traj=4,
    fname_traj="./trajectories/traj_%03d.npy",
    fname_time="./trajectories/time.npy",
    fname_weights="./trajectories/weight_%03d.npy",
    fname_forcing="./trajectories/forcing_%03d.pkl",
    fname_derivs="./trajectories/deriv_%03d.npy",
    dtype=np.float64,
)
data = TrainingData(pool, which_trajs=[0, 1, 2, 3],
                    percent_time_length=1.0, leggauss_deg=5, nsave_rom=15)

# 2. Initial bases from POD; oblique projection (Psi = Phi here => orthogonal)
U, _, _ = compute_POD(pool, normalize=True)
Phi = U[:, :2]
projection = LinearProjection([Phi, Phi])

# 3. Latent quadratic model  zdot = A z + H:zz^T (+ B u)
model = PolynomialModel(r=2, poly_comp=[1, 2], dtype=np.float64)

# 4. Train bases and operators jointly, on their natural manifolds
registry = ParamRegistry(model, projection)
rom = NitromModule(data, registry, fom=my_fom, n_substeps=15)
rom.set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])
train(rom, n_epochs=400, lr=1.0, optimizer_type="lbfgs", tol=1e-14)
```

Run in parallel over trajectories with `mpiexec -n 4 python train.py` (NumPy backend)
or `torchrun --nproc_per_node=4 train.py` (PyTorch backend) — the cost and gradient are
all-reduced automatically and every rank runs the identical optimizer in lockstep.

## What each step does

1. **Data** — {class}`~nitrom.training_data.TrainingPool` loads trajectory snapshots,
   per-trajectory weights, time derivatives, and forcing callables from disk and shards
   them across ranks. {class}`~nitrom.training_data.TrainingData` is the view handed to
   an optimizer: it selects trajectories, truncates them, and sets up Gauss–Legendre
   quadrature for the adjoint integrals.
2. **Projection** — {class}`~nitrom.projections.LinearProjection` is an oblique
   projection with trial basis $\Phi$ and test basis $\Psi$. Starting from POD modes
   ({func}`~nitrom.utils.compute_POD`) is the usual warm start.
3. **Model** — {class}`~nitrom.latent_space_models.PolynomialModel` implements
   $f(z,u) = A_1 z + A_2 : zz^\top + \dots + Bu$ with analytic Jacobians, adjoint
   right-hand sides, and parameter VJPs.
4. **Training** — {class}`~nitrom.roms.ParamRegistry` unifies model and projection
   parameters into a single optimization vector;
   {class}`~nitrom.optimization.NitromModule` evaluates the trajectory-mismatch cost by
   time-marching the ROM and differentiates it with the adjoint;
   {func}`~nitrom.optimization.train` drives Adam, SGD, or L-BFGS, retracting
   manifold-typed parameters on their Grassmann/Stiefel manifolds.

For the pure operator-inference problem, {func}`~nitrom.optimization.solve_opinf` skips
iteration entirely and returns the exact regularized weighted-least-squares solution.
