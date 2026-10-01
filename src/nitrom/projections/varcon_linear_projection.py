"""Variationally consistent (symplectic Petrov-Galerkin) linear projection."""

from typing import Any

from ..backend import get_backend
from ..utils import validate_symplectic_structure
from .linear_projection import LinearProjection


def _is_sparse(J: Any) -> bool:
    """True when ``J`` is a scipy sparse matrix."""
    import scipy.sparse as sp

    return sp.issparse(J)


class VarconLinearProjection(LinearProjection):
    r"""
    Variationally consistent (symplectic Petrov-Galerkin) linear projection:
    the trial basis is :math:`\Phi = U` and the test basis is **derived** as
    :math:`\Psi = J U`, where :math:`J` is the (skew-symmetric) full-order
    symplectic structure.  The resulting operators,

    .. math::

        \text{encode}(q) = (J U)^\top q, \qquad
        \text{decode}(z) = U\, (U^\top J^\top U)^{-1} z,

    are exactly the :class:`LinearProjection` maps with bases
    ``[U, J @ U]``, and satisfy
    :math:`\text{encode} \circ \text{decode} = \mathrm{Id}`.

    Only ``Phi`` is a free parameter (``param_names == ["Phi"]``):
    :math:`\Psi` is recomputed from every update, and the VJPs chain the
    :math:`\Psi` gradient back through :math:`\delta\Psi = J\,\delta\Phi`,

    .. math::

        \nabla_\Phi = \nabla_\Phi^{\text{direct}} + J^\top\, \nabla_\Psi .

    Pairs naturally with
    :class:`~nitrom.latent_space_models.hamiltonian_polynomial_model.HamiltonianPolynomialModel`,
    which also exposes ``"Phi"``:
    :class:`~nitrom.roms.param_registry.ParamRegistry` then shares the basis
    between the two and sums their gradients.

    :param U: trial basis of shape ``(n, r)``
    :param J: full-order symplectic structure of shape ``(n, n)``, skew
        symmetric with ``n`` even; a dense array, or a scipy sparse matrix
        on the numpy backend
    """

    def __init__(self, U: Any, J: Any):
        bkend = get_backend()

        if _is_sparse(J):
            if bkend.is_torch:
                raise TypeError(
                    "sparse J requires the numpy backend; densify J or "
                    "set_backend('numpy')."
                )
        else:
            J = bkend.asarray(J, dtype=U.dtype, device=bkend.device_of(U))
        # Canonical-type structure: J = -J^T and J @ J = -I (so J^{-1} = -J
        # and J^{-T} = J).  The reduced Psi^T Phi = U^T J^T U is only skew.
        validate_symplectic_structure(J)
        n = J.shape[0]

        if U.ndim != 2 or U.shape[0] != n:
            raise ValueError(
                f"U must have shape ({n}, r), got {tuple(U.shape)}."
            )
        if U.shape[1] % 2 != 0:
            raise ValueError(
                f"U must have an even number of columns: the reduced matrix "
                f"Psi^T Phi = U^T J^T U is skew-symmetric and hence singular "
                f"for odd r, got r={U.shape[1]}."
            )

        self.J = J
        super().__init__([U, self._applyJ(U)])
        # Psi is derived, so only Phi is registered as a free parameter.
        self._param_names = ["Phi"]

    # ------------------------------------------------------------------
    # J application (dense backend array or scipy sparse)
    # ------------------------------------------------------------------
    def _applyJ(self, X: Any) -> Any:
        """``J @ X`` for dense or scipy-sparse ``J``."""
        bkend = get_backend()
        if _is_sparse(self.J):
            return bkend.asarray(
                self.J @ bkend.to_numpy(X), dtype=X.dtype,
                device=bkend.device_of(X),
            )
        return self.J @ X

    def _applyJT(self, X: Any) -> Any:
        """``J^T @ X`` for dense or scipy-sparse ``J``."""
        bkend = get_backend()
        if _is_sparse(self.J):
            return bkend.asarray(
                self.J.T @ bkend.to_numpy(X), dtype=X.dtype,
                device=bkend.device_of(X),
            )
        return self.J.T @ X

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @staticmethod
    def canonical_J(
        n: int, dtype: Any = None, device: str = "cpu", sparse: bool = False
    ) -> Any:
        r"""
        Build the canonical symplectic structure

        .. math::

            J = \begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix}
            \in \mathbb{R}^{n \times n},

        with ``n`` even.

        :param n: ambient dimension (must be even)
        :param dtype: data type; defaults to the backend's ``float64``
        :param device: device for array allocation (dense only)
        :param sparse: if ``True``, return a scipy CSR matrix (numpy backend
            only)
        :rtype: backend array or scipy sparse matrix
        """
        if n % 2 != 0:
            raise ValueError(f"canonical J needs an even dimension, got n={n}.")
        half = n // 2
        if sparse:
            import scipy.sparse as sp

            eye = sp.identity(half, format="csr")
            return sp.bmat([[None, eye], [-eye, None]], format="csr")
        bkend = get_backend()
        dtype = dtype if dtype is not None else bkend.float64
        J = bkend.zeros((n, n), dtype=dtype, device=device)
        eye = bkend.eye(half, dtype=dtype, device=device)
        J[:half, half:] = eye
        J[half:, :half] = -eye
        return J

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def get_params(self) -> list[Any]:
        """Return ``[Phi]`` (``Psi = J @ Phi`` is derived)."""
        return [self.Phi]

    def update(self, params: list) -> None:
        r"""
        Update the trial basis and recompute :math:`\Psi = J\,\Phi` and
        :math:`S = (\Psi^\top \Phi)^{-1}`.

        :param params: list holding the single tensor ``Phi`` of shape
            ``(N, r)``
        :type params: list
        """
        Phi = params[0]
        super().update([Phi, self._applyJ(Phi)])

    # ------------------------------------------------------------------
    # VJPs: chain grad_Psi back to Phi through Psi = J Phi
    # ------------------------------------------------------------------
    def vjp_encode(self, q: Any, v: Any) -> tuple:
        r"""
        VJP of the encoder w.r.t. ``Phi``, chaining through
        :math:`\Psi = J\,\Phi`:

        .. math::

            \nabla_\Phi = \nabla_\Phi^{\text{direct}} + J^\top \nabla_\Psi .

        :param q: full-space vector of shape ``(N,)`` or ``(m, N)``
        :param v: upstream adjoint seed of shape ``(r,)`` or ``(m, r)``
        :returns: the 1-tuple ``(grad_Phi,)``
        :rtype: tuple
        """
        grad_Phi, grad_Psi = super().vjp_encode(q, v)
        return (grad_Phi + self._applyJT(grad_Psi),)

    def vjp_decode(self, z: Any, v: Any) -> tuple:
        r"""
        VJP of the decoder w.r.t. ``Phi``, chaining through
        :math:`\Psi = J\,\Phi` (see :meth:`vjp_encode`).

        :param z: reduced-space vector of shape ``(r,)`` or ``(m, r)``
        :param v: upstream adjoint seed of shape ``(N,)`` or ``(m, N)``
        :returns: the 1-tuple ``(grad_Phi,)``
        :rtype: tuple
        """
        grad_Phi, grad_Psi = super().vjp_decode(z, v)
        return (grad_Phi + self._applyJT(grad_Psi),)
