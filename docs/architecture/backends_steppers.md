# Backends and time steppers

Two cross-cutting services: an array-backend shim that lets the same math run on NumPy
or PyTorch, and a time-stepping layer with a discrete adjoint.

## The backend shim

All numerical code is written against a thin {class}`~nitrom.backend.Backend`
abstraction. Select the implementation once per process:

```python
from nitrom.backend import set_backend
set_backend("numpy")   # or "torch"
```

Objects (models, projections, modules) bind to whichever backend is active **at
construction time**. On NumPy you get a light CPU stack with optional MPI trajectory
parallelism; on PyTorch you get autograd, GPU execution, and
`torch.distributed` parallelism. `setup_distributed()` / `cleanup_distributed()` wrap
process-group setup (NCCL on CUDA, gloo otherwise) for `torchrun` jobs, and the
`mpi_*` helpers reduce across MPI ranks.

A deliberate design rule keeps imports cheap: **`torch` and `mpi4py` are imported
lazily, inside functions** — a NumPy-only session never pulls in torch, and neither is
needed to import `nitrom`.

## General-purpose solvers

{func}`~nitrom.time_steppers.solve_ivp` integrates batched IVPs with `rk2`, `rk4`,
`backward_euler` (Newton with batched Jacobians), adaptive `rk45`
(Dormand–Prince 5(4)), or the symplectic schemes below, returning the solution
interpolated onto the requested evaluation times;
{func}`~nitrom.time_steppers.solve_ivp_dense` returns it on the integration grid
instead. {func}`~nitrom.time_steppers.solve_adjoint_ivp_discrete` marches the adjoint
state backward through the *discrete* RK stages and accumulates parameter VJPs — this
is what makes NiTROM's discrete gradients exactly consistent with the forward
integrator.

Every scheme is a plain Butcher tableau, so the Newton stage solve, dense output, and
discrete adjoint are scheme-agnostic. The symplectic **implicit midpoint** (1-stage
Gauss) and **Yoshida-4** (its triple-jump composition, written as a 3-stage diagonally
implicit tableau) entered as new tableaus with no new solver code, with verified
convergence orders 2.00 and 4.00.

## Symplectic integration of linear systems

For linear Hamiltonian dynamics $\dot z = Az$, `nitrom.time_steppers.linear_symplectic`
provides the exact Cayley propagator

$$C(h) = \left(I - \tfrac{h}{2} A\right)^{-1}\left(I + \tfrac{h}{2} A\right),$$

dense on both backends and sparse via `splu` on NumPy, plus a Yoshida-4 composition of
it. {func}`~nitrom.time_steppers.linear_symplectic_solve` integrates with it, and the
backward adjoint march reuses the transposed propagators. Symplecticity
$M^\top J M = J$ is pinned in `tests/time_steppers/`.
