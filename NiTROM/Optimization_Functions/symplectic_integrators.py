"""
Symplectic time integration for (possibly nonseparable) Hamiltonian systems.

The module is organized as a four-tier abstraction ladder:

    1. ``implicit_midpoint_step``   — one implicit-midpoint step.
    2. ``yoshida4_step``            — one 4th-order step (Yoshida triple-jump of 1).
    3. ``symplectic_integrate``     — march through a prescribed time grid using 2.
    4. ``symplectic_solve``         — integrate at internal ``dt`` and return the
                                      solution interpolated to an arbitrary
                                      ``t_eval`` grid (the user-facing solver).

Each tier is useful in isolation: callers that want to audit symplecticity should work
with tier 3 (no interpolation), or call tier 4 with ``return_internal=True`` to also
receive the (non-interpolated) state on the internal integration grid.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def implicit_midpoint_step(
    rhs,
    jacobian,
    z_n: np.ndarray,
    h: float,
    *,
    newton_tol: float = 1.0e-12,
    newton_maxiter: int = 20,
) -> np.ndarray:
    """
    Advance one implicit-midpoint step for z' = rhs(z).

    The implicit midpoint rule is symplectic for general Hamiltonian systems.
    """
    z_n = np.asarray(z_n, dtype=float)
    z_np1 = z_n + h * np.asarray(rhs(z_n), dtype=float)
    eye = np.eye(z_n.size)

    for _ in range(newton_maxiter):
        z_mid = 0.5 * (z_n + z_np1)
        residual = z_np1 - z_n - h * np.asarray(rhs(z_mid), dtype=float)

        if np.linalg.norm(residual, ord=np.inf) <= newton_tol:
            return z_np1

        jac_mid = np.asarray(jacobian(z_mid), dtype=float)
        system = eye - 0.5 * h * jac_mid
        delta = np.linalg.solve(system, -residual)
        z_np1 = z_np1 + delta

        if np.linalg.norm(delta, ord=np.inf) <= newton_tol:
            return z_np1

    raise RuntimeError(
        "Implicit midpoint Newton solve did not converge. "
        "Consider reducing the timestep or improving the Jacobian."
    )


def yoshida4_step(
    rhs,
    jacobian,
    z_n: np.ndarray,
    h: float,
    *,
    newton_tol: float = 1.0e-12,
    newton_maxiter: int = 20,
) -> np.ndarray:
    """
    Advance one fourth-order symplectic step for a nonseparable Hamiltonian system.

    Uses Yoshida's triple-jump composition of ``implicit_midpoint_step``.
    """
    cbrt2 = np.cbrt(2.0)
    gamma_1 = 1.0 / (2.0 - cbrt2)
    gamma_2 = -cbrt2 / (2.0 - cbrt2)

    z_np1 = np.asarray(z_n, dtype=float)
    for coeff in (gamma_1, gamma_2, gamma_1):
        z_np1 = implicit_midpoint_step(
            rhs,
            jacobian,
            z_np1,
            coeff * h,
            newton_tol=newton_tol,
            newton_maxiter=newton_maxiter,
        )

    return z_np1


def symplectic_integrate(
    rhs,
    jacobian,
    z0: np.ndarray,
    time: np.ndarray,
    *,
    newton_tol: float = 1.0e-12,
    newton_maxiter: int = 20,
) -> np.ndarray:
    """
    Integrate a Hamiltonian system on a prescribed time grid with a fourth-order
    symplectic method (Yoshida). Works for nonseparable Hamiltonians.

    Parameters
    ----------
    rhs
        Callable returning dz/dt at a state z.
    jacobian
        Callable returning d(rhs)/dz at a state z.
    z0
        Initial state.
    time
        Monotone increasing array of times where the solution is stored.
        Each consecutive gap is taken as one symplectic step.
    """
    time = np.asarray(time, dtype=float)
    z0 = np.asarray(z0, dtype=float)

    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")
    if len(time) < 2:
        raise ValueError("At least two time points are required.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time grid must be strictly increasing.")

    Z = np.zeros((z0.size, len(time)), dtype=float)
    Z[:, 0] = z0

    for k in range(1, len(time)):
        h = time[k] - time[k - 1]
        Z[:, k] = yoshida4_step(
            rhs,
            jacobian,
            Z[:, k - 1],
            h,
            newton_tol=newton_tol,
            newton_maxiter=newton_maxiter,
        )

    return Z


def symplectic_solve(
    rhs,
    jacobian,
    z0: np.ndarray,
    t_eval: np.ndarray,
    *,
    dt: float | None = None,
    return_internal: bool = False,
    newton_tol: float = 1.0e-12,
    newton_maxiter: int = 20,
):
    """
    Integrate z' = rhs(z) at internal step ``dt`` and return the solution
    at the user-requested times ``t_eval``.

    Parameters
    ----------
    dt
        Internal symplectic step. If ``None``, integrates directly on ``t_eval``
        (no interpolation).
    return_internal
        If ``True``, also return ``(fine_time, Z_fine)``: the internal
        integration grid and the (non-interpolated, symplecticity-preserving)
        state on that grid. Use these for energy / invariant audits.

    Returns
    -------
    Z_eval : ndarray of shape (n, len(t_eval))
        State at ``t_eval`` (cubic-spline-interpolated when ``dt is not None``).
    fine_time, Z_fine : optional
        Returned only if ``return_internal=True``.
    """
    t_eval = np.asarray(t_eval, dtype=float)

    if dt is None:
        Z = symplectic_integrate(
            rhs, jacobian, z0, t_eval,
            newton_tol=newton_tol, newton_maxiter=newton_maxiter,
        )
        if return_internal:
            return Z, t_eval, Z
        return Z

    t0, tf = float(t_eval[0]), float(t_eval[-1])
    fine_time = np.arange(t0, tf + 0.5 * dt, dt)
    if fine_time[-1] < tf:
        fine_time = np.append(fine_time, tf)

    Z_fine = symplectic_integrate(
        rhs, jacobian, z0, fine_time,
        newton_tol=newton_tol, newton_maxiter=newton_maxiter,
    )
    Z_eval = CubicSpline(fine_time, Z_fine, axis=1)(t_eval)

    if return_internal:
        return Z_eval, fine_time, Z_fine
    return Z_eval
