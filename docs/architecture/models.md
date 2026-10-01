# Latent-space models

A model is the latent dynamics $\dot z = f(z, u)$ together with everything the adjoint
needs: the Jacobian-transpose (adjoint RHS) and the parameter VJPs.

## The contract

{class}`~nitrom.latent_space_models.Model` requires:

| Method | Role |
| --- | --- |
| `evaluate_rhs(t, z)` | the latent vector field $f$ |
| `evaluate_adjoint_rhs(t, z, Z)` | adjoint dynamics $-(\partial f/\partial z)^\top \lambda$ |
| `vjp_evaluate_rhs(z, v, …)` | VJPs of $f$ w.r.t. every parameter |
| `get_params()` / `update_params(…)` | expose and replace parameters, in `param_names` order |

## The unconstrained workhorse

{class}`~nitrom.latent_space_models.PolynomialModel` implements
$f(z,u) = A_1 z + A_2 : zz^\top + \dots + Bu$ for any set of polynomial degrees
(`poly_comp`), with analytic Jacobians, adjoint RHS, and parameter VJPs. Every
structured model below ultimately delegates to one of these.

## The structured branch

{class}`~nitrom.latent_space_models.StructuredPolynomialModel` is the abstract
delegation skeleton: a structured model holds **free parameters**, assembles them into
the operator tensors of an **inner** `PolynomialModel`, and chains gradients from the
inner tensors back to the free parameters. Subclasses implement three things: the
parameter specs, `assemble_inner_tensors()`, and `project_inner_gradients()`.

{class}`~nitrom.latent_space_models.GasPolynomialModel`
: Free variables $(K, R, Q, S)$ assemble operators that are **globally asymptotically
  stable by construction**: $A = ((K-K^\top) - R^{-1}R^{-\top})\tilde{Q}$ and
  $H_{ijk} = (S_{ilk}-S_{lik})\tilde{Q}_{lj}$ with $\tilde{Q}=Q^{-1}Q^{-\top}$.
  `retract_general_tensors_to_gas_tensors` maps an existing (possibly unstable)
  $(A,H)$ onto this manifold via a Lyapunov solve, so a Galerkin or OpInf model can
  warm-start GAS training.

{class}`~nitrom.latent_space_models.AtrPolynomialModel`
: The attracting-trapping-region variant — for full-order models whose attractor is
  *not* the origin, where making the origin globally attracting is the wrong prior.
  Same API as the GAS model.

{class}`~nitrom.latent_space_models.HamiltonianPolynomialModel`
: The latent dynamics derive from a polynomial **reduced Hamiltonian**
  $\hat{H}(z) = \tfrac12 z^\top A_2 z + \tfrac13 A_3(z,z,z) + \cdots$, as
  $\dot z = \hat{J}\nabla\hat{H}(z)$ with $\hat{J} = \Phi^\top J \Phi$. The assembly
  map is $T_k = \hat{J}\,\mathrm{Sym}(A_{k+1})$, and the gradient chain rule includes
  a $\Phi$ contribution through $\hat{J}$ — which is why this model *shares* its
  `"Phi"` parameter with {class}`~nitrom.projections.VarconLinearProjection` through
  the registry. Autonomous (no forcing); parameter order $[A_2, A_3, \Phi]$. The
  derivation is in {doc}`../theory`.

## Extension recipe

For a new operator constraint, subclass
{class}`~nitrom.latent_space_models.StructuredPolynomialModel`: declare free
parameters, write the assembly onto inner polynomial tensors, and write its adjoint
(the chain rule back). RHS/adjoint/VJP machinery comes from the inner model. Pin the
gradient chain against finite differences — `tests/models/` has the pattern, including
a cubic-only case that isolates the slot-symmetrization adjoint.
