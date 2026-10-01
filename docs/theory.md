# Theory

This page collects the objectives NiTROM optimizes and the structure-preserving
extensions of the `hamiltonian-v2` line of development. Notation: full-order state
$x \in \mathbb{R}^N$, latent state $z \in \mathbb{R}^r$, trial/test bases
$\Phi, \Psi \in \mathbb{R}^{N\times r}$.

## Objectives

### Operator inference

Project the data, then fit the latent operators by regularized weighted least squares
on the projected derivatives:

$$
J_{\mathrm{OpInf}} \;=\; \sum_i w_i\,\bigl\lVert \dot z_i - f(t_i, z_i)\bigr\rVert^2
\;+\; \lambda \sum_k \lVert A_k \rVert^2 ,
$$

with $f(z,u) = A_1 z + A_2 : zz^\top + \dots + Bu$. This is quadratic in the operators,
so {func}`~nitrom.optimization.solve_opinf` returns the exact solution without
iteration; {class}`~nitrom.optimization.OpInfModule` exposes the same cost with
analytic gradients when you want to iterate (e.g. with frozen or manifold-typed
parameters).

### Trajectory-based training (NiTROM)

Operator inference never time-marches the model it fits. NiTROM instead penalizes the
mismatch of the **integrated** ROM outputs,

$$
J_{\mathrm{NiTROM}} \;=\; \sum_j \alpha_j^{-1} \sum_i
\bigl\lVert y^{(j)}(t_i) - \hat y^{(j)}(t_i) \bigr\rVert^2 ,
$$

where $\hat y^{(j)}$ comes from actually solving $\dot z = f(z,u)$,
$z_0 = \mathrm{encode}(x_0)$, and decoding. The gradient with respect to *all*
parameters — operators **and** bases — is computed by the adjoint method, either
discretely (back through the RK stages, exactly consistent with the forward
integrator) or continuously. Bases are optimized on their natural matrix manifolds
(Grassmann/Stiefel) with retractions and vector transport.

### Nonlinear-manifold reconstruction

For a {class}`~nitrom.projections.PolynomialProjection`, the decoder coefficients are
fit by reconstruction error,

$$
J_{\mathrm{man}} \;=\; \sum_i w_i\,\bigl\lVert x_i - \mathrm{decode}(z_i)\bigr\rVert^2
\;+\; \lambda \sum_k \lVert A_k \rVert^2 ,
$$

({class}`~nitrom.optimization.PolyManifoldInfModule`), or the manifold is trained
jointly inside the NiTROM objective.

## Hamiltonian ROMs

### Setting

Let the full-order system carry a canonical-type symplectic structure
$J = -J^\top = J^{-\top}$, $JJ = -I$ (validated at construction by
{func}`~nitrom.utils.validate_symplectic_structure`). With trial basis $\Phi$, the
latent dynamics derive from a polynomial **reduced** Hamiltonian:

$$
\hat{H}(z) \;=\; \tfrac12\, z^\top A_2 z \;+\; \tfrac13\, A_3(z,z,z) \;+\;\cdots,
\qquad
\dot z \;=\; \hat{J}\,\nabla \hat{H}(z),
\qquad
\hat{J} \;=\; \Phi^\top J \Phi .
$$

Conceptually $\hat{H} = H \circ \mathrm{decode}$, but it is parameterized directly by
the latent coefficient tensors $A_2, A_3, \dots$, trained jointly with $\Phi$. The
reduced $\hat{J}$ is non-canonical and only skew. Both $\hat{H}$ and $\nabla\hat{H}$
use the same $1/k$ coefficient convention.

{class}`~nitrom.latent_space_models.HamiltonianPolynomialModel` realizes this as a
structured model: it assembles the inner polynomial tensors
$T_k = \hat{J}\,\mathrm{Sym}(A_{k+1})$ and chains gradients back,
$\nabla_{A_{k+1}} = \mathrm{Sym}^\top(\hat{J}^\top G_k)$ and
$\nabla_\Phi = J\Phi\, \nabla_{\hat{J}}^{\top} + J^\top\!\Phi\, \nabla_{\hat{J}}$
from $\delta\hat{J} = \delta\Phi^\top(J\Phi) + (\Phi^\top J)\delta\Phi$, where $G_k$
are the discrete-adjoint gradients with respect to the assembled tensors.

### Variationally consistent projection

{class}`~nitrom.projections.VarconLinearProjection` is a
{class}`~nitrom.projections.LinearProjection` with bases $[U,\ JU]$: encoder
$(JU)^\top$, decoder $U(U^\top J^\top U)^{-1}$, so
$\mathrm{encode}\circ\mathrm{decode} = \mathrm{Id}$. Only `"Phi"` $\,(= U)$ is free;
$\Psi = JU$ is derived, and $r$ must be even since $U^\top J^\top U$ is skew. The
registry shares $\Phi$ between projection and model automatically.

### The shared-basis gradient

The training gradient follows from the Lagrangian (one trajectory shown,
$U \equiv \Phi$):

$$
\mathcal{L}(U, A_2, A_3)
=
\underbrace{\textstyle\sum_{ij}\mathcal{J}_{ij}}_{\text{(a) misfit}}
+
\underbrace{\int_{t_0}^{t_i} \lambda^\top\bigl(\dot z
    - \hat{J}\,\nabla\hat{H}(z)\bigr)\,\mathrm{d}t}_{\text{(b) dynamics}}
+
\underbrace{\lambda_0^\top\bigl(z_0 - (JU)^\top x_0\bigr)}_{\text{(c) initial condition}} .
$$

$U$ appears in exactly three places, and the registry sums the three labeled
contributions:

$$
\nabla_U \mathcal{L}
=
\underbrace{\nabla^{\mathrm{dec}}_{\Phi}
    + J^\top \nabla^{\mathrm{dec}}_{\Psi}}_{\text{(a) reconstruction (decoder)}}
+
\underbrace{J\Phi\, \nabla_{\hat{J}}^{\top}
    + J^\top \Phi\, \nabla_{\hat{J}}}_{\text{(b) dynamics through } \hat{J}}
+
\underbrace{J^\top \nabla^{\mathrm{enc}}_{\Psi}}_{\text{(c) ICs (encoder)}},
\qquad
\nabla_{\hat{J}} = \sum_k \bigl\langle G_k,\,
    \mathrm{Sym}(A_{k+1}) \bigr\rangle_{\mathrm{trailing}} .
$$

The $\nabla_\Psi$ partials are an artifact of the parameterization, not extra physics:
the generic linear projection treats $(\Phi, \Psi)$ as independent and returns the
pair of partials, and the varcon subclass imposes $\Psi = J\Phi$ via the chain rule
$d\mathcal{L}/dU = \partial\mathcal{L}/\partial\Phi|_\Psi
+ J^\top\,\partial\mathcal{L}/\partial\Psi|_\Phi$. The encoder term reproduces (c)
exactly, and the decoder's two chained partials sum to $d/dU$ of
$U(U^\top J^\top U)^{-1}$ — an identity pinned against autograd in the projection
tests.

### Symplectic integrators

Implicit midpoint is the 1-stage Gauss tableau; Yoshida-4 is its triple-jump
composition with substeps $(\gamma_1, \gamma_2, \gamma_1)$, written as a 3-stage
diagonally implicit RK tableau. Because both are plain Butcher tableaus, the existing
Newton stage solve, dense solver, and discrete adjoint work unchanged (verified orders
2.00 and 4.00). For linear dynamics the exact Cayley propagator
$C(h) = (I - \tfrac{h}{2}A)^{-1}(I + \tfrac{h}{2}A)$ is available, including a sparse
`splu` path on NumPy; the backward adjoint march uses the transposed propagators.
Tests verify symplecticity $M^\top J M = J$, exact conservation of quadratic
Hamiltonians over long horizons, and bounded (non-secular) energy drift for cubic
ones.

## References

- A. Padovan, B. Vollmer, D. J. Bodony, *Data-driven model reduction via non-intrusive
  optimization of projection operators and reduced-order dynamics* (the NiTROM paper).
- B. Peherstorfer, K. Willcox, *Data-driven operator inference for nonintrusive
  projection-based model reduction*, Computer Methods in Applied Mechanics and
  Engineering, 2016.
- L. Peng, K. Mohseni, *Symplectic model reduction of Hamiltonian systems*, SIAM
  Journal on Scientific Computing, 2016.
- H. Yoshida, *Construction of higher order symplectic integrators*, Physics Letters
  A, 1990.
