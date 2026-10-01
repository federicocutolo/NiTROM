"""Abstract base for polynomial ROMs with a structured (constraint-preserving) parameterization."""

import abc
from typing import Any

from .model import Model
from .polynomial_model import PolynomialModel


class StructuredPolynomialModel(Model, metaclass=abc.ABCMeta):
    r"""
    Abstract base for polynomial ROMs with a *structured* parameterization.

    A structured model holds free parameters (e.g. the GAS factors ``K``,
    ``R``, ``Q``, ``S``, or Hamiltonian coefficient tensors) and assembles
    them into the operator tensors of an internal
    :class:`~nitrom.latent_space_models.polynomial_model.PolynomialModel`,
    to which all RHS, adjoint, and inner VJP evaluations are delegated.
    Gradients are accumulated w.r.t. the assembled *inner* tensors and mapped
    back to the free parameters by :meth:`project_inner_gradients` (a
    hand-derived chain rule), applied once at the end of an adjoint sweep.

    Subclasses implement three hooks:

    * :meth:`_param_specs` -- names, shapes and initializations of the free
      parameters;
    * :meth:`assemble_inner_tensors` -- build the inner operator tensors from
      the current free parameters;
    * :meth:`project_inner_gradients` -- chain rule from inner-tensor
      gradients back to the free parameters;

    and may override two more:

    * :meth:`_inner_poly_comp` -- polynomial degrees of the inner model
      (identity by default);
    * :meth:`_precompute` -- cache assembly intermediates, called after every
      parameter update (no-op by default).

    :param r: reduced state dimension
    :type r: int
    :param poly_comp: polynomial degrees, e.g. ``[1, 2]``
    :type poly_comp: list[int]
    :param device: device for array allocation (ignored by the NumPy backend)
    :type device: str
    :param dtype: data type for arrays; defaults to the backend's ``float64``
    :type dtype: backend dtype or None
    :param instability_threshold: norm threshold for blow-up guard
    :type instability_threshold: float
    :param params: optional list of initial free-parameter tensors, ordered
        as :attr:`param_names`.  If ``None``, parameters are initialized from
        their :meth:`_param_specs` entries.
    :type params: list or None
    :param forcing_config: optional dict with keys ``"forcing_exists"``
        (bool) and ``"m"`` (int).  See :class:`PolynomialModel`.
    :type forcing_config: dict or None
    """

    def __init__(
        self,
        r: int,
        poly_comp: list[int],
        device: str = "cpu",
        dtype: Any = None,
        instability_threshold: float = 1e6,
        params: list | None = None,
        forcing_config: dict | None = None,
    ):
        # Determine the free-parameter names, shapes, and initializations
        specs = self._param_specs(r, poly_comp)
        param_names: list[str] = [name for name, _, _ in specs]

        # Track forcing.  An optional fixed input operator may be supplied via
        # ``forcing_config["B"]``; B stays a parameter but is flagged
        # non-learnable so that its gradient is zeroed.
        forcing_exists = forcing_config is not None and forcing_config.get(
            "forcing_exists", False
        )
        B_fixed = forcing_config.get("B") if forcing_config else None
        if forcing_exists:
            param_names.append("B")

        super().__init__(r, param_names, device, dtype)
        bkend = self.backend
        self.poly_comp = self._inner_poly_comp(poly_comp)
        self.forcing_exists = forcing_exists

        # Set the free parameters as attributes
        if params is not None:
            for name, tensor in zip(param_names, params, strict=True):
                setattr(
                    self, name,
                    bkend.asarray(tensor, dtype=self.dtype, device=self.device),
                )
        else:
            # Initialize the free parameters (B handled below)
            for name, shape, init in specs:
                alloc = bkend.zeros if init == "zeros" else bkend.randn
                setattr(
                    self, name,
                    alloc(shape, dtype=self.dtype, device=self.device),
                )
            # Initialize B (fixed value if supplied, else zeros)
            if forcing_exists:
                m = forcing_config["m"]
                self.B = (
                    bkend.asarray(B_fixed, dtype=self.dtype, device=self.device)
                    if B_fixed is not None
                    else bkend.zeros((r, m), dtype=self.dtype, device=self.device)
                )

        # A supplied fixed B always overrides any value from params.
        if forcing_exists and B_fixed is not None:
            self.B = bkend.asarray(B_fixed, dtype=self.dtype, device=self.device)

        # Precompute assembly intermediates
        self._precompute()

        # Assemble inner tensors and create the inner PolynomialModel
        tensors = self.assemble_inner_tensors()
        self.model = PolynomialModel(
            r,
            self.poly_comp,
            device=device,
            dtype=self.dtype,
            instability_threshold=instability_threshold,
            tensors=tensors,
            forcing_config=forcing_config,
        )

    # ------------------------------------------------------------------
    # Parameterization hooks
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _param_specs(
        self, r: int, poly_comp: list[int]
    ) -> list[tuple[str, tuple[int, ...], str]]:
        r"""
        Specification of the free parameters (everything except ``B``).

        Each entry is ``(name, shape, init)`` where ``init`` is ``"randn"``
        or ``"zeros"``, used when no initial parameters are supplied.

        :param r: reduced state dimension
        :param poly_comp: polynomial degrees requested by the caller
        :rtype: list[tuple[str, tuple[int, ...], str]]
        """
        ...

    @staticmethod
    def _inner_poly_comp(poly_comp: list[int]) -> list[int]:
        """
        Polynomial degrees carried by the inner :class:`PolynomialModel`.

        Identity by default; subclasses override it when the assembled
        dynamics need extra degrees (e.g. the ATR model, which adds a
        constant term).
        """
        return list(poly_comp)

    def _precompute(self) -> None:
        """Cache assembly intermediates.  Called on construction and after
        every :meth:`update_params`; no-op by default."""

    @abc.abstractmethod
    def assemble_inner_tensors(self) -> list[Any]:
        r"""
        Build the inner operator tensors from the current free parameters.

        :returns: list of tensors matching the inner
            :class:`PolynomialModel` param order, with ``B`` appended when
            forcing is present.
        :rtype: list
        """
        ...

    @abc.abstractmethod
    def project_inner_gradients(self, inner_grads: list[Any]) -> list[Any]:
        r"""
        Chain rule from gradients w.r.t. the assembled inner tensors back to
        the free parameters.

        :param inner_grads: gradients w.r.t. :meth:`inner_params`
        :type inner_grads: list
        :returns: gradients matching :attr:`param_names`
        :rtype: list
        """
        ...

    # ------------------------------------------------------------------
    # Generic machinery
    # ------------------------------------------------------------------
    def get_params(self) -> list[Any]:
        """Return the current free-parameter tensors as a list."""
        return [getattr(self, name) for name in self.param_names]

    def update_params(self, params: list) -> None:
        r"""
        Update the free parameters (and B if present), reassemble the inner
        tensors, and push them into the inner :class:`PolynomialModel`.

        :param params: parameter tensors matching :attr:`param_names`
        :type params: list
        """
        for name, tensor in zip(self.param_names, params, strict=True):
            setattr(self, name, tensor)
        self._precompute()
        self.model.update_params(self.assemble_inner_tensors())

    def evaluate_rhs(self, t: float, z: Any, **kwargs) -> Any:
        """Delegate to the inner :class:`PolynomialModel`."""
        return self.model.evaluate_rhs(t, z, **kwargs)

    def evaluate_adjoint_rhs(self, t: float, z: Any, Z: Any, **kwargs) -> Any:
        """Delegate to the inner :class:`PolynomialModel`."""
        return self.model.evaluate_adjoint_rhs(t, z, Z, **kwargs)

    def inner_params(self) -> list[Any]:
        """Return the assembled inner operator tensors."""
        return self.model.get_params()

    def inner_vjp_evaluate_rhs(
        self, z: Any, v: Any, reg: float = 0.0, **kwargs
    ) -> list[Any]:
        r"""
        VJP of the RHS with respect to the inner model tensors.

        Calls the inner :class:`PolynomialModel` VJP.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :param v: upstream adjoint seed, same shape as ``z``
        :returns: list of gradients matching :meth:`inner_params`
        :rtype: list
        """
        return self.model.vjp_evaluate_rhs(z, v, reg=reg, **kwargs)

    def inner_batched_vjp_evaluate_rhs(
        self, Z: Any, V: Any, U: Any = None, out: list | None = None,
        max_bytes: int = 64 << 20,
    ) -> list[Any]:
        """Batched VJP w.r.t. the inner tensors.

        The free parameters are recovered from these by
        :meth:`project_inner_gradients`, which the caller applies once at the
        end of the sweep.
        """
        return self.model.batched_vjp_evaluate_rhs(
            Z, V, U=U, out=out, max_bytes=max_bytes
        )

    def vjp_evaluate_rhs(
        self, z: Any, v: Any, reg: float = 0.0, **kwargs
    ) -> list[Any]:
        r"""
        VJP of the RHS with respect to the free parameters.

        :param z: state vector of shape ``(n,)`` or ``(m, n)``
        :param v: upstream adjoint seed, same shape as ``z``
        :returns: list of gradients matching :attr:`param_names`
        :rtype: list
        """
        inner_grads = self.inner_vjp_evaluate_rhs(z, v, reg=reg, **kwargs)
        return self.project_inner_gradients(inner_grads)
