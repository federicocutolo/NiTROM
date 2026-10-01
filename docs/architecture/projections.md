# Projections

A projection is the pair of maps between the ambient space $\mathbb{R}^N$ and the
latent space $\mathbb{R}^r$, *plus* the vector–Jacobian products the adjoint method
needs to differentiate through them.

## The contract

{class}`~nitrom.projections.Projection` binds to the active backend at construction
and requires:

| Method | Role |
| --- | --- |
| `encode(q)` | ambient → latent |
| `decode(z)` | latent → ambient |
| `vjp_encode(q, v, …)` | parameter VJPs of the encoder (used at initial conditions) |
| `vjp_decode(z, v, …)` | parameter VJPs of the decoder (used in the misfit) |
| `vjp_decode_state(z, v, …)` | VJP of the decoder w.r.t. the latent state |
| `get_params()` / `update(…)` | expose and replace parameters, in `param_names` order |

Anything satisfying this contract can be trained jointly with a model — the registry
and the inference modules never look inside.

## Concrete projections

{class}`~nitrom.projections.LinearProjection`
: Oblique projection with **independent** trial and test bases:
  $\mathrm{encode}(q)=\Psi^\top q$ and
  $\mathrm{decode}(z)=\Phi(\Psi^\top\Phi)^{-1}z$. Setting $\Psi=\Phi$ recovers the
  orthogonal (Galerkin) case. Its VJPs return the pair of partials
  $(\partial\mathcal{L}/\partial\Phi|_\Psi,\ \partial\mathcal{L}/\partial\Psi|_\Phi)$.

{class}`~nitrom.projections.VarconLinearProjection`
: Variationally consistent (symplectic Petrov–Galerkin) projection: the trial basis is
  $\Phi = U$ and the test basis is **derived**, $\Psi = JU$, where $J$ is the
  full-order symplectic structure. Only `"Phi"` is a free parameter; the subclass
  applies the chain rule $\delta\Psi = J\,\delta U$, folding every $\Psi$-partial back
  onto $U$ as $J^\top \nabla_\Psi$. The latent dimension $r$ must be **even**
  ($U^\top J^\top U$ is skew, hence singular for odd $r$). This is the projection that
  pairs with the Hamiltonian model — see {doc}`../theory` for the full shared-basis
  gradient.

{class}`~nitrom.projections.PolynomialProjection`
: Linear encoder with a **polynomial-manifold decoder**,
  $\mathrm{decode}(z)=\Phi S z + \mathbb{P}\sum_k A_k z^{\otimes k}$, where the
  complementary projector $\mathbb{P} = I - \Phi S\Psi^\top$ guarantees
  $\mathrm{encode}(\mathrm{decode}(z))=z$ exactly. Trained with
  {class}`~nitrom.optimization.PolyManifoldInfModule` (reconstruction loss) or jointly
  inside the NiTROM objective.

## Extension recipe

Subclass {class}`~nitrom.projections.Projection` directly for a genuinely new
encoder/decoder pair, or {class}`~nitrom.projections.LinearProjection` when you are
constraining its bases (the varcon class is ~100 lines: constructor + chained VJPs).
Validate shapes in the constructor, and pin your VJPs against finite differences or
torch autograd the way `tests/projections/` does.
