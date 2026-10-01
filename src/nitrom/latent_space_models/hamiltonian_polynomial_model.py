"""Hamiltonian-constrained polynomial ROM whose dynamics derive from a polynomial reduced Hamiltonian."""

from string import ascii_lowercase
from typing import Any

import numpy as np

from ..backend import get_backend
from ..utils import validate_symplectic_structure
from .structured_polynomial_model import StructuredPolynomialModel


def _is_sparse(J: Any) -> bool:
    """True when ``J`` is a scipy sparse matrix."""
    import scipy.sparse as sp

    return sp.issparse(J)


class HamiltonianPolynomialModel(StructuredPolynomialModel):
    r"""
    Hamiltonian-constrained polynomial ROM.

    The latent dynamics derive from a polynomial **reduced** Hamiltonian
    :math:`\hat{H}` -- conceptually :math:`\hat{H} = H \circ \mathrm{decode}`,
    the full-order Hamiltonian composed with the decoder, but parameterized
    here directly by free latent-space coefficient tensors rather than by
    that composition.  For ``poly_comp = [1, 2, ...]`` (right-hand-side
    degrees, as for the other polynomial models) the free coefficient
    tensors are :math:`A_2` of shape ``(r, r)``, :math:`A_3` of shape
    ``(r, r, r)``, ... (tensor order = RHS degree + 1), with

    .. math::

        \hat{H}(z) = \sum_k \frac{1}{k+1}\, A_{k+1}(z, \ldots, z)
             = \tfrac12\, z^\top A_2 z + \tfrac13\, A_3(z, z, z) + \cdots,
        \qquad
        \dot{z} = \hat{J}\, \nabla \hat{H}(z),
        \qquad
        \hat{J} = \Phi^\top J \Phi,

    where :math:`J` is the (skew-symmetric) full-order symplectic structure
    and :math:`\Phi` the trial basis.  ``Phi`` is a **free parameter** of the
    model (listed last in :attr:`param_names`), so
    :class:`~nitrom.roms.param_registry.ParamRegistry` automatically shares
    it with a projection that also exposes ``"Phi"`` (e.g.
    :class:`~nitrom.projections.varcon_linear_projection.VarconLinearProjection`)
    and sums their gradients; :math:`\hat{J}` is recomputed from ``Phi`` on
    every :meth:`update_params`.

    **Assembled inner tensors.**  The gradient of the degree-``p`` term is
    the slot-symmetrized contraction

    .. math::

        \nabla \Bigl[\tfrac{1}{p} A_p(z, \ldots, z)\Bigr]
            = \mathrm{Sym}(A_p)(\cdot, z, \ldots, z),
        \qquad
        \mathrm{Sym}(A_p) = \frac{1}{p} \sum_{s=1}^{p}
            \text{(move slot $s$ to the front)},

    e.g. :math:`\mathrm{Sym}(A_2) = \tfrac12 (A_2 + A_2^\top)` and
    :math:`\mathrm{Sym}(A_3)_{ljk} = \tfrac13 (A_{3,ljk} + A_{3,jlk} +
    A_{3,jkl})`.  The inner
    :class:`~nitrom.latent_space_models.polynomial_model.PolynomialModel`
    tensor of RHS degree :math:`k` is therefore

    .. math::

        T_k = \hat{J}\cdot_1 \mathrm{Sym}(A_{k+1}),

    (contraction over the first slot).

    **Gradients** (:meth:`project_inner_gradients`).  With :math:`G_k` the
    accumulated gradient w.r.t. :math:`T_k`,

    .. math::

        \nabla_{A_{k+1}} &= \mathrm{Sym}^\top\!\bigl(\hat{J}^\top \cdot_1
            G_k\bigr), \\
        \nabla_{\hat{J}} &= \sum_k \langle G_k, \mathrm{Sym}(A_{k+1})
            \rangle_{\text{trailing}}, \\
        \nabla_{\Phi} &= J \Phi\, \nabla_{\hat{J}}^\top
            + J^\top \Phi\, \nabla_{\hat{J}},

    where :math:`\mathrm{Sym}^\top` sums the inverse slot permutations, the
    trailing contraction leaves the leading index pair free, and the last
    line follows from :math:`\delta\hat{J} = \delta\Phi^\top (J\Phi) +
    (\Phi^\top J)\,\delta\Phi`.

    Regularization (``reg``) applied by the optimization modules targets the
    **assembled** quadratic tensor :math:`T_2`, not :math:`A_3`;
    :meth:`project_inner_gradients` maps its gradient back to :math:`A_3`
    and :math:`\Phi` consistently.

    Forcing is not supported: the dynamics are autonomous Hamiltonian.

    :param r: reduced state dimension
    :type r: int
    :param poly_comp: RHS polynomial degrees, e.g. ``[1]`` (quadratic H) or
        ``[1, 2]`` (quadratic + cubic H); degree 0 is not allowed
    :type poly_comp: list[int]
    :param J: full-order symplectic structure of shape ``(n, n)`` with ``n``
        even, canonical-type: :math:`J = -J^\top` and :math:`JJ = -I`
        (equivalently :math:`J^{-1} = -J`, :math:`J^{-\top} = J`); a dense
        array, or a scipy sparse matrix on the numpy backend.  The reduced
        :math:`\hat{J} = \Phi^\top J \Phi` is non-canonical and only
        guaranteed skew-symmetric.
    :param Phi: trial basis of shape ``(n, r)``
    :param device: device for array allocation (ignored by the NumPy backend)
    :type device: str
    :param dtype: data type for arrays; defaults to the backend's ``float64``
    :type dtype: backend dtype or None
    :param instability_threshold: norm threshold for blow-up guard
    :type instability_threshold: float
    :param ham_params: optional list of initial coefficient tensors
        ``[A2, A3, ...]`` (without ``Phi``, which always comes from the
        ``Phi`` argument).  If ``None``, coefficients are initialized
        randomly.
    :type ham_params: list or None
    """

    def __init__(
        self,
        r: int,
        poly_comp: list[int],
        J: Any,
        Phi: Any,
        device: str = "cpu",
        dtype: Any = None,
        instability_threshold: float = 1e6,
        ham_params: list | None = None,
    ):
        if any(k < 1 for k in poly_comp):
            raise ValueError(
                f"HamiltonianPolynomialModel needs RHS degrees >= 1 "
                f"(a constant term has no Hamiltonian), got {poly_comp}."
            )

        bkend = get_backend()
        dtype_eff = dtype if dtype is not None else bkend.float64

        # -- validate J ---------------------------------------------------
        if _is_sparse(J):
            if bkend.is_torch:
                raise TypeError(
                    "sparse J requires the numpy backend; densify J or "
                    "set_backend('numpy')."
                )
        else:
            J = bkend.asarray(J, dtype=dtype_eff, device=device)
        # Canonical-type structure: J = -J^T and J @ J = -I (so J^{-1} = -J
        # and J^{-T} = J).  The reduced Jhat = Phi^T J Phi is non-canonical
        # and only guaranteed skew.
        validate_symplectic_structure(J)
        n = J.shape[0]

        # -- validate Phi -------------------------------------------------
        if Phi.ndim != 2 or Phi.shape[0] != n or Phi.shape[1] != r:
            raise ValueError(
                f"Phi must have shape ({n}, {r}), got {tuple(Phi.shape)}."
            )

        # Needed by _param_specs and _precompute, which the base __init__
        # calls before this constructor resumes.
        self._J = J
        self._n = n

        if ham_params is not None:
            if len(ham_params) != len(poly_comp):
                raise ValueError(
                    f"ham_params must hold one coefficient tensor per entry "
                    f"of poly_comp ({len(poly_comp)}), got {len(ham_params)}."
                )
            coeffs = list(ham_params)
        else:
            coeffs = [
                bkend.randn((r,) * (k + 1), dtype=dtype_eff, device=device)
                for k in poly_comp
            ]
        params = [
            *coeffs,
            bkend.asarray(Phi, dtype=dtype_eff, device=device),
        ]

        super().__init__(
            r,
            poly_comp,
            device=device,
            dtype=dtype,
            instability_threshold=instability_threshold,
            params=params,
            forcing_config=None,
        )

    # ------------------------------------------------------------------
    # StructuredPolynomialModel hooks
    # ------------------------------------------------------------------
    def _param_specs(
        self, r: int, poly_comp: list[int]
    ) -> list[tuple[str, tuple[int, ...], str]]:
        """Coefficient tensors ``A{k+1}`` per RHS degree ``k``, then ``Phi``
        last (so the registry order is coefficients first, ``Phi`` last)."""
        specs: list[tuple[str, tuple[int, ...], str]] = [
            (f"A{k + 1}", (r,) * (k + 1), "randn") for k in poly_comp
        ]
        specs.append(("Phi", (self._n, r), "randn"))
        return specs

    def _applyJ(self, X: Any) -> Any:
        """``J @ X`` for dense or scipy-sparse ``J`` (numpy backend only for
        sparse)."""
        bkend = self.backend
        if _is_sparse(self._J):
            return bkend.asarray(
                self._J @ bkend.to_numpy(X), dtype=self.dtype, device=self.device
            )
        return self._J @ X

    def _applyJT(self, X: Any) -> Any:
        """``J^T @ X`` for dense or scipy-sparse ``J``."""
        bkend = self.backend
        if _is_sparse(self._J):
            return bkend.asarray(
                self._J.T @ bkend.to_numpy(X),
                dtype=self.dtype, device=self.device,
            )
        return self._J.T @ X

    def _precompute(self) -> None:
        r"""Recompute :math:`\hat{J} = \Phi^\top J \Phi` from the current
        ``Phi`` (runs on construction and after every parameter update)."""
        self._JPhi = self._applyJ(self.Phi)
        self.Jhat = self.Phi.T @ self._JPhi

    @staticmethod
    def _slot_permutations(p: int) -> list[tuple[int, ...]]:
        """Axis orders that move each slot of an order-``p`` tensor to the
        front, preserving the order of the remaining slots."""
        return [
            (s, *(a for a in range(p) if a != s)) for s in range(p)
        ]

    def _sym(self, A: Any) -> Any:
        r"""Slot symmetrization :math:`\mathrm{Sym}(A_p) = \frac{1}{p}
        \sum_s \text{(move slot s to the front)}`."""
        bkend = self.backend
        p = A.ndim
        if p == 2:
            return 0.5 * (A + A.T)
        out = None
        for axes in self._slot_permutations(p):
            term = bkend.permute(A, axes)
            out = term if out is None else out + term
        return out / p

    def _sym_adjoint(self, G: Any) -> Any:
        """Adjoint of :meth:`_sym`: the same average over the *inverse* slot
        permutations."""
        bkend = self.backend
        p = G.ndim
        if p == 2:
            return 0.5 * (G + G.T)
        out = None
        for axes in self._slot_permutations(p):
            inv = tuple(np.argsort(axes))
            term = bkend.permute(G, inv)
            out = term if out is None else out + term
        return out / p

    def assemble_inner_tensors(self) -> list[Any]:
        r"""
        Build the inner RHS tensors :math:`T_k = \hat{J} \cdot_1
        \mathrm{Sym}(A_{k+1})` from the current coefficients and ``Phi``.

        The symmetrized coefficients are cached in ``self._sym_coeffs`` for
        reuse by :meth:`project_inner_gradients`.

        :rtype: list
        """
        bkend = self.backend
        tensors = []
        self._sym_coeffs = []
        for k in self.poly_comp:
            A = getattr(self, f"A{k + 1}")
            S = self._sym(A)
            self._sym_coeffs.append(S)
            rest = ascii_lowercase[1 : k + 1]  # trailing (uncontracted) slots
            tensors.append(
                bkend.einsum(f"az,z{rest}->a{rest}", self.Jhat, S)
            )
        return tensors

    def project_inner_gradients(self, inner_grads: list[Any]) -> list[Any]:
        r"""
        Chain rule from gradients w.r.t. the assembled tensors :math:`T_k`
        back to the Hamiltonian coefficients and ``Phi`` (see the class
        docstring for the formulas).

        :param inner_grads: gradients w.r.t. :meth:`inner_params`
        :type inner_grads: list
        :returns: ``[grad_A2, grad_A3, ..., grad_Phi]``
        :rtype: list
        """
        bkend = self.backend

        grads = []
        grad_Jhat = None
        for i, k in enumerate(self.poly_comp):
            G = inner_grads[i]
            rest = ascii_lowercase[1 : k + 1]

            # grad w.r.t. Sym(A): Jhat^T applied to the leading index, then
            # the adjoint slot-symmetrization back to A.
            G_sym = bkend.einsum(f"az,a{rest}->z{rest}", self.Jhat, G)
            grads.append(self._sym_adjoint(G_sym))

            # grad_Jhat_{al} = sum_rest G_{a rest} Sym(A)_{l rest}
            term = bkend.einsum(
                f"a{rest},z{rest}->az", G, self._sym_coeffs[i]
            )
            grad_Jhat = term if grad_Jhat is None else grad_Jhat + term

        # grad_Phi through Jhat = Phi^T J Phi:
        #   dJhat = dPhi^T (J Phi) + (Phi^T J) dPhi
        #   => grad_Phi = (J Phi) grad_Jhat^T + (J^T Phi) grad_Jhat
        grad_Phi = self._JPhi @ grad_Jhat.T + self._applyJT(self.Phi) @ grad_Jhat
        grads.append(grad_Phi)

        return grads

    # ------------------------------------------------------------------
    # Hamiltonian diagnostics
    # ------------------------------------------------------------------
    def compute_hamiltonian(self, z: Any) -> Any:
        r"""
        Evaluate the reduced Hamiltonian

        .. math::

            \hat{H}(z) = \sum_k \frac{1}{k+1}\, A_{k+1}(z, \ldots, z)

        :param z: state of shape ``(r,)`` or batched ``(m, r)``
        :returns: scalar, or batch of scalars ``(m,)``
        :rtype: backend array
        """
        bkend = self.backend
        H = None
        for k in self.poly_comp:
            A = getattr(self, f"A{k + 1}")
            sub = ascii_lowercase[: k + 1]
            operands = ",".join(f"...{c}" for c in sub)
            term = bkend.einsum(f"{sub},{operands}->...", A, *([z] * (k + 1)))
            term = term / (k + 1)
            H = term if H is None else H + term
        return H

    def compute_hamiltonian_gradient(self, z: Any) -> Any:
        r"""
        Evaluate :math:`\nabla \hat{H}(z) = \sum_k \mathrm{Sym}(A_{k+1})(\cdot,
        z, \ldots, z)`.

        :param z: state of shape ``(r,)`` or batched ``(m, r)``
        :rtype: backend array
        """
        bkend = self.backend
        g = None
        for i, k in enumerate(self.poly_comp):
            S = self._sym_coeffs[i]
            rest = ascii_lowercase[1 : k + 1]
            operands = ",".join(f"...{c}" for c in rest)
            term = bkend.einsum(f"a{rest},{operands}->...a", S, *([z] * k))
            g = term if g is None else g + term
        return g
