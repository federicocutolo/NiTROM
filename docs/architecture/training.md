# Training

One contract — a scalar cost with an analytic gradient — lets a single training loop
drive every inference problem in the package.

## ParamRegistry: one optimization vector

{class}`~nitrom.roms.ParamRegistry` unifies the parameters of a model and a projection
into a single flat parameter list. Each parameter is registered once by **name**; a
name appearing in *both* objects (e.g. a basis `"Phi"` that enters the decoder *and*
the latent dynamics, as in the Hamiltonian ROM) is marked **shared**: one variable
feeds both sites and the two gradient contributions are summed. Freezing and manifold
typing below operate on these registered names.

## The module contract

{class}`~nitrom.optimization.InferenceModule` (built on the minimal `BackendModule`
parameter container) requires exactly two methods:

| Method | Role |
| --- | --- |
| `forward()` | scalar training cost |
| `gradient()` | its **analytic** gradient, one entry per parameter |

Common controls available on any module:

- `set_unlearnable("B")` / `set_learnable(…)` — freeze parameters (gradients zeroed),
  e.g. to keep a fixed input operator $B = \Phi^\top B_{\mathrm{fom}}$;
- `set_manifold_types(["Phi", "Psi"], ["grassmann", "stiefel"])` — optimize bases on
  matrix manifolds instead of Euclidean space.

## The three concrete modules

| Module | Cost |
| --- | --- |
| {class}`~nitrom.optimization.OpInfModule` | $J = \sum_i w_i\lVert \dot z_i - f(t_i,z_i)\rVert^2 + \lambda\sum_k\lVert A_k\rVert^2$ — least-squares fit of the projected derivatives |
| {class}`~nitrom.optimization.PolyManifoldInfModule` | $J = \sum_i w_i\lVert x_i - \mathrm{decode}(z_i)\rVert^2 + \lambda\sum_k\lVert A_k\rVert^2$ — nonlinear-manifold reconstruction error |
| {class}`~nitrom.optimization.NitromModule` | $J = \sum_j \alpha_j^{-1}\sum_i \lVert y^{(j)}(t_i) - \hat y^{(j)}(t_i)\rVert^2$ — output mismatch of the **time-marched** ROM, differentiated by the adjoint |

`NitromModule` takes a {class}`~nitrom.roms.ParamRegistry` (so it can train model and
projection jointly, with shared parameters) and supports both the **discrete** adjoint
(exactly consistent with the forward integrator) and the **continuous** adjoint,
selected via `adjoint_method`.

## train(): one loop for everything

{func}`~nitrom.optimization.train` runs Adam, SGD, or L-BFGS with optional restarts
and multi-start perturbations, recording `loss_history` / `gradnorm_history`:

- **PyTorch backend** — `torch.optim` optimizers, LR schedulers, distributed
  all-reduce of cost and gradient;
- **NumPy backend** — each optimizer supplies only a search *direction*; the step
  comes from a strong-Wolfe line search with Armijo backtracking
  (`nitrom.optimization.manifold_optimization` provides the Riemannian L-BFGS);
- **either backend** — manifold-typed parameters are retracted after each step, their
  gradients tangent-projected, and momentum buffers vector-transported between tangent
  spaces.

Under `mpiexec` / `torchrun`, every rank evaluates its shard of trajectories, cost and
gradient are all-reduced, and all ranks run the identical optimizer in lockstep.

## solve_opinf(): skip the iteration

For the pure operator-inference problem, {func}`~nitrom.optimization.solve_opinf`
returns the exact regularized weighted-least-squares solution directly — honouring
frozen parameters — instead of iterating. Use it as the fast baseline, or as a warm
start for NiTROM training.
