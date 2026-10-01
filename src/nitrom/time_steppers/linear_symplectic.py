r"""
Symplectic time integration specialized for **linear** systems
:math:`\dot{z} = A z` with a constant matrix :math:`A`.

Fast companion to the ``"implicit_midpoint"`` / ``"yoshida4"`` Butcher
tableaus in :mod:`nitrom.time_steppers.time_stepper`: for a linear RHS the
implicit-midpoint update is the exact Cayley transform

.. math::

    \left(I - \tfrac{h}{2} A\right) z_{n+1}
        = \left(I + \tfrac{h}{2} A\right) z_n
    \quad\Longleftrightarrow\quad
    z_{n+1} = C(h)\, z_n,
    \qquad
    C(h) = \left(I - \tfrac{h}{2} A\right)^{-1}
           \left(I + \tfrac{h}{2} A\right),

so no Newton iteration is needed and constant operators are precomputed once
per (unique) step size.  The 4th-order Yoshida step is the fixed composition
:math:`C(\gamma_1 h)\, C(\gamma_2 h)\, C(\gamma_1 h)`, applied via a single
reusable one-step operator.

This module is a *library utility* invoked explicitly by callers that know
their dynamics are linear (e.g. a linear Hamiltonian ROM,
:math:`A = \hat{J}(A_2 + A_2^\top)`): it needs the constant matrix ``A``,
which does not fit the ``f(t, x)`` right-hand-side contract of the general
tableau machinery.  :class:`~nitrom.optimization.modules.nitrom_module.NitromModule`
keeps the tableau path -- the tableaus plus
:func:`~nitrom.time_steppers.time_stepper.solve_adjoint_ivp_discrete` are the
general mechanism, and this module is the O(1)-per-step fast path.  For a
linear Hamiltonian system it reproduces the tableau integrator's trajectory
to Newton tolerance, only far cheaper.

Dense vs sparse ``A``:

* **Dense** ``A`` (numpy or torch backend): the one-step propagator
  :math:`M = C(\gamma_1 h)\, C(\gamma_2 h)\, C(\gamma_1 h)` is formed
  explicitly and the march is one matmul per step.  Best for the small dense
  reduced operators.
* **Sparse** (scipy) ``A`` -- numpy backend only: ``M`` is never formed (it
  would densify).  The two distinct denominators
  :math:`I - \gamma_i \tfrac{h}{2} A` are LU-factorized once (scipy ``splu``)
  and each step is 3 sparse mat-vecs + 3 back-substitutions.  Best for large
  sparse ``A`` (e.g. an uncompressed FOM).  scipy ``splu`` has no torch
  counterpart, so a sparse ``A`` on the torch backend raises ``TypeError``;
  densify ``A`` or ``set_backend("numpy")``.

The backward adjoint marches use the **transposed** propagators: the adjoint
dynamics :math:`\dot{\lambda} = -A^\top \lambda` stepped by the same scheme
give :math:`C_{\mathrm{adj}}(h) = C(h)^{-\top}`, so marching :math:`\lambda`
*backward* over a step applies :math:`C(h)^\top` -- see :func:`interval_maps`.
"""
from typing import Any

from nitrom.backend import get_backend
from nitrom.utils import interp_quadratic

from .time_stepper import _YOSHIDA_G1, _YOSHIDA_G2


def _is_sparse(A: Any) -> bool:
    """True when ``A`` is a scipy sparse matrix."""
    import scipy.sparse as sp

    return sp.issparse(A)


def _check_sparse_backend(A: Any) -> None:
    """Reject scipy-sparse operators on the torch backend."""
    if _is_sparse(A) and get_backend().is_torch:
        raise TypeError(
            "sparse A requires the numpy backend (scipy splu has no torch "
            "counterpart); densify A or set_backend('numpy')."
        )


def cayley_step(A: Any, h: float) -> Any:
    r"""
    Dense implicit-midpoint one-step propagator for :math:`\dot{z} = A z`:

    .. math::

        C(h) = \left(I - \tfrac{h}{2} A\right)^{-1}
               \left(I + \tfrac{h}{2} A\right)

    (symplectic for a linear Hamiltonian ``A``).

    :param A: dense system matrix of shape ``(n, n)`` (backend array)
    :param h: step size
    :returns: the propagator ``C(h)`` of shape ``(n, n)``
    :rtype: backend array
    """
    bkend = get_backend()
    eye = bkend.eye(A.shape[0], dtype=A.dtype, device=bkend.device_of(A))
    return bkend.solve(eye - (0.5 * h) * A, eye + (0.5 * h) * A)


def yoshida4_propagator(A: Any, h: float) -> Any:
    r"""
    Dense 4th-order symplectic one-step propagator for :math:`\dot{z} = A z`
    (Yoshida triple jump of the Cayley step):
    :math:`z_{n+1} = C(\gamma_1 h)\, C(\gamma_2 h)\, C(\gamma_1 h)\, z_n`.

    :param A: dense system matrix of shape ``(n, n)`` (backend array)
    :param h: step size
    :rtype: backend array
    """
    c1 = cayley_step(A, _YOSHIDA_G1 * h)
    c2 = cayley_step(A, _YOSHIDA_G2 * h)
    return c1 @ c2 @ c1  # apply C1, then C2, then C1


def linear_propagator(A: Any, tau: float, dt: float | None = None) -> Any:
    r"""
    Compounded symplectic propagator over a **duration** ``tau`` for
    :math:`\dot{z} = A z`: the single matrix equal to marching
    ``floor(tau/dt)`` Yoshida steps of size ``dt`` plus one fractional step
    for the remainder,

    .. math::

        P(\tau) = M(\mathrm{rem})\, M(dt)^{n},
        \qquad n = \lfloor \tau / dt \rfloor,
        \quad \mathrm{rem} = \tau - n\, dt,

    so ``z(t + tau) = P(tau) @ z(t)`` in one matvec.  All factors are
    rational functions of the same ``A``, so they commute and the composition
    order is immaterial.  Being a product of symplectic maps, ``P`` is
    symplectic.

    Built once (``log2(n)`` matmuls by repeated squaring) and reused; this
    replaces step-by-step sub-stepping wherever only the state at ``t + tau``
    is needed.  ``dt=None`` (or ``dt >= tau``) means a single step of size
    ``tau``.  Dense ``A`` only.

    :param A: dense system matrix of shape ``(n, n)`` (backend array)
    :param tau: duration to propagate over
    :param dt: internal Yoshida step size (``None`` for a single step)
    :rtype: backend array
    """
    if dt is None or dt >= tau:
        return yoshida4_propagator(A, tau)
    bkend = get_backend()
    n_full = int((tau / dt) + 1e-9)  # tolerate roundoff in tau/dt
    rem = tau - n_full * dt
    P = bkend.xp.linalg.matrix_power(yoshida4_propagator(A, dt), n_full)
    if rem > 1e-9 * dt:  # skip a zero-length step
        P = yoshida4_propagator(A, rem) @ P
    return P


def _sparse_cayley_ops(A: Any, h: float) -> tuple:
    """Factor the h-Cayley step for sparse ``A``: return (LU of
    ``(I - h/2 A)``, ``(I + h/2 A)`` for mat-vec).  Applying the step is
    ``lu.solve(B_plus @ z)``."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    n = A.shape[0]
    ident = sp.identity(n, format="csc")
    minus = (ident - 0.5 * h * A).tocsc()
    plus = (ident + 0.5 * h * A).tocsr()
    return splu(minus), plus


def _build_stepper(A: Any, h: float):
    """Return a callable ``step(z)`` advancing one 4th-order symplectic step
    for :math:`\\dot{z} = A z`, with all h-dependent operators precomputed.
    ``z`` may be a vector ``(n,)`` or a stack of columns ``(n, B)``.  Sparse
    ``A`` uses LU + back-substitution (no dense propagator); dense ``A`` uses
    the explicit propagator."""
    if _is_sparse(A):
        A = A.tocsc()
        # The two outer C(g1 h) share one factorization; the inner C(g2 h)
        # gets its own.
        lu1, plus1 = _sparse_cayley_ops(A, _YOSHIDA_G1 * h)
        lu2, plus2 = _sparse_cayley_ops(A, _YOSHIDA_G2 * h)

        def step(z):
            z = lu1.solve(plus1 @ z)  # C(g1 h)
            z = lu2.solve(plus2 @ z)  # C(g2 h)
            z = lu1.solve(plus1 @ z)  # C(g1 h)
            return z

        return step

    M = yoshida4_propagator(A, h)  # dense: one matmul per step
    return lambda z: M @ z


def linear_symplectic_integrate(A: Any, z0: Any, time: Any) -> Any:
    r"""
    Integrate :math:`\dot{z} = A z` on a prescribed time grid with the
    4th-order symplectic (Yoshida) method.  Each consecutive gap is one step.
    The step operator is built once per distinct gap (once total for a
    uniform grid).

    :param A: system matrix of shape ``(n, n)``; dense backend array, or a
        scipy sparse matrix on the numpy backend
    :param z0: initial condition of shape ``(n,)`` or batched ``(B, n)``
    :param time: strictly increasing time grid of shape ``(n_t,)``, at least
        two points
    :returns: trajectory of shape ``(n, n_t)`` or ``(B, n, n_t)``
    :rtype: backend array
    """
    _check_sparse_backend(A)
    bkend = get_backend()

    if hasattr(time, "ndim") and time.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")
    n_t = len(time)
    if n_t < 2:
        raise ValueError("At least two time points are required.")
    gaps = [float(time[k + 1] - time[k]) for k in range(n_t - 1)]
    if any(g <= 0.0 for g in gaps):
        raise ValueError("time grid must be strictly increasing.")

    batched = z0.ndim == 2
    dev = bkend.device_of(z0)
    dtype = z0.dtype

    if batched:
        B, n = z0.shape
        Z = bkend.zeros((B, n, n_t), dtype=dtype, device=dev)
        Z[:, :, 0] = z0
    else:
        n = z0.shape[0]
        Z = bkend.zeros((n, n_t), dtype=dtype, device=dev)
        Z[:, 0] = z0

    sparse = _is_sparse(A)

    if sparse:
        # Sparse march: LU back-substitutions on numpy column stacks.
        def _make_step(gap: float):
            return _build_stepper(A, gap)

        def _apply(step, z):
            z_np = bkend.to_numpy(z)
            out = step(z_np.T if batched else z_np)
            return bkend.asarray(
                out.T if batched else out, dtype=dtype, device=dev
            )
    else:
        # Dense march: one propagator matmul per step (z @ M^T when the
        # batched states are stored as rows).
        def _make_step(gap: float):
            return yoshida4_propagator(A, gap)

        def _apply(M, z):
            return z @ bkend.permute(M, (1, 0)) if batched else M @ z

    # One stepper per distinct gap (one total on a uniform grid).
    cache: dict[float, Any] = {}
    for k in range(1, n_t):
        key = round(gaps[k - 1], 15)
        step = cache.get(key)
        if step is None:
            step = _make_step(gaps[k - 1])
            cache[key] = step
        Z[..., k] = _apply(step, Z[..., k - 1])
    return Z


def linear_symplectic_solve(
    A: Any,
    z0: Any,
    t_eval: Any,
    *,
    dt: float | None = None,
    return_internal: bool = False,
):
    r"""
    Integrate :math:`\dot{z} = A z` at internal step ``dt`` and return the
    solution at the requested times ``t_eval`` (interpolated with
    :func:`~nitrom.utils.interp_quadratic` when ``dt`` subdivides the grid).
    Takes the constant matrix ``A`` (dense backend array, or scipy sparse on
    the numpy backend) instead of an ``f(t, x)`` right-hand side.

    :param A: system matrix of shape ``(n, n)``
    :param z0: initial condition of shape ``(n,)`` or batched ``(B, n)``
    :param t_eval: requested times, shape ``(n_eval,)``
    :param dt: internal symplectic step.  If ``None``, integrates directly on
        ``t_eval``.
    :param return_internal: if ``True``, also return ``(fine_time, Z_fine)``
        on the internal grid
    :returns: ``Z_eval`` of shape ``(n, n_eval)`` or ``(B, n, n_eval)``, or
        the tuple ``(Z_eval, fine_time, Z_fine)``
    """
    _check_sparse_backend(A)
    bkend = get_backend()

    if dt is None:
        Z = linear_symplectic_integrate(A, z0, t_eval)
        return (Z, t_eval, Z) if return_internal else Z

    t0, tf = float(t_eval[0]), float(t_eval[-1])
    dev = bkend.device_of(z0)
    dtype = z0.dtype
    n_fine = int((tf - t0) / dt + 0.5) + 1
    fine_time = t0 + dt * bkend.arange(n_fine, dtype=dtype, device=dev)
    if float(fine_time[-1]) < tf - 1e-9 * dt:
        fine_time = bkend.concatenate(
            [fine_time, bkend.asarray([tf], dtype=dtype, device=dev)], axis=0
        )

    Z_fine = linear_symplectic_integrate(A, z0, fine_time)

    # Fast path: when every requested time coincides with an internal knot
    # (commensurate dt, the recommended setup), return the integrator states
    # directly -- exact, and skips the interpolation stencils entirely.
    idx = [round((float(te) - t0) / dt) for te in t_eval]
    on_knots = (
        min(idx) >= 0
        and max(idx) < len(fine_time)
        and max(
            abs(float(fine_time[i]) - float(te))
            for i, te in zip(idx, t_eval, strict=True)
        )
        <= 1e-3 * dt
    )
    if on_knots:
        idx_arr = bkend.asarray(idx, dtype=None, device=dev)
        if get_backend().is_torch:
            idx_arr = idx_arr.long()
        Z_eval = Z_fine[..., idx_arr]
    else:
        Z_eval = interp_quadratic(
            bkend.asarray(t_eval, dtype=dtype, device=dev), fine_time, Z_fine
        )

    return (Z_eval, fine_time, Z_fine) if return_internal else Z_eval


def interval_maps(
    A: Any,
    gap: float,
    offsets: Any,
    dt: float | None = None,
) -> tuple[Any, list[Any], list[Any]]:
    r"""
    Propagators for one snapshot interval ``[t, t + gap]``: the full-gap
    propagator ``M_gap`` and, for each interior offset :math:`d_i` (e.g. a
    Gauss-Legendre node mapped to the interval), the node propagators
    ``N_i`` (``t -> t + d_i``) and ``Q_i`` (``t + d_i -> t + gap``), so that

    .. math::

        z(t + d_i) = N_i\, z(t), \qquad z(t + \mathrm{gap}) = Q_i\, z(t + d_i),
        \qquad M_{\mathrm{gap}} = Q_i N_i .

    The exact discrete adjoint of this march uses the **transposes**: with a
    terminal adjoint :math:`\lambda(t + \mathrm{gap})`,

    .. math::

        \lambda(t) = M_{\mathrm{gap}}^\top\, \lambda(t + \mathrm{gap}),
        \qquad
        \lambda(t + d_i) = Q_i^\top\, \lambda(t + \mathrm{gap}),

    because each Cayley factor of the backward march of
    :math:`\dot{\lambda} = -A^\top \lambda` is the transpose of the
    corresponding forward factor.

    Dense ``A`` only.

    :param A: dense system matrix of shape ``(n, n)`` (backend array)
    :param gap: interval length
    :param offsets: interior offsets :math:`0 \le d_i \le \mathrm{gap}`
        (iterable of floats)
    :param dt: internal Yoshida step for the compounded propagators
    :returns: ``(M_gap, N_list, Q_list)``
    """
    M_gap = linear_propagator(A, gap, dt)
    N = [linear_propagator(A, float(d), dt) for d in offsets]
    Q = [linear_propagator(A, gap - float(d), dt) for d in offsets]
    return M_gap, N, Q
