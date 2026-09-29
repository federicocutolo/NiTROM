"""
Gaussian-pulse initial-condition sampler for the wave-equation example.

Single source of truth for HOW Gaussian-pulse trajectories are drawn. The
parameter ranges themselves are owned by the case (wave_cases.sample_ranges),
so the training set (Dirichlet/main.py) and the held-out test set
(test_nitrom.py) are drawn from exactly the same box by construction. Vary only
the ``seed`` to get a different, non-overlapping draw from the same space.

``ranges`` is a dict with keys "amp", "sigma", "x0", each a (lo, hi) tuple,
already resolved per-case (the sigma floor scales with grid resolution dx). Two
optional keys select the boundary/momentum regime:
  * "periodic" (bool): wrap the pulse with a minimal-image distance so any x0 in
    [0, L) gives a smooth periodic IC (Gruber-Tezaur). Default False (Dirichlet).
  * "momentum" ("traveling" | "zero"): "traveling" sets the matching momentum so
    the pulse is one running wave (Dirichlet); "zero" sets p=0 so the bump splits
    into TWO counter-propagating waves (periodic Gruber-Tezaur). Default
    "traveling".
"""
import numpy as np


def gaussian_ic(fom, amp, width, x0, direction, periodic_L=None):
    """Gaussian displacement pulse with the matching exact momentum field.

    ``direction`` in {-1, +1} sets a single running wave; ``direction == 0``
    gives zero momentum, so the bump splits into two counter-propagating waves.
    ``periodic_L`` (the domain length) wraps the pulse via the minimal-image
    distance so an x0 near a boundary stays smooth across the periodic seam.
    """
    d = fom.x - x0
    if periodic_L is not None:
        d = (d + periodic_L / 2) % periodic_L - periodic_L / 2   # minimal image
    q = amp * np.exp(-d ** 2 / (2 * width ** 2))
    p = direction * fom.c * (d / width ** 2) * q
    return np.concatenate([q, p])


def draw_params(n_traj, rng, ranges):
    """Sample (amp, sigma, mu, direction) for ``n_traj`` ICs from ``ranges``.

    The draw order (amp, sigma, mu, [direction]) is part of the contract: it
    fixes the RNG stream, so a given seed always yields the same trajectories.
    For ``momentum == "zero"`` the direction is fixed at 0 (no draw consumed).
    """
    zero_momentum = ranges.get("momentum", "traveling") == "zero"
    params = []
    for _ in range(n_traj):
        amp       = rng.uniform(*ranges["amp"])
        sigma     = rng.uniform(*ranges["sigma"])
        mu        = rng.uniform(*ranges["x0"])
        direction = 0 if zero_momentum else rng.choice([-1, 1])
        params.append((amp, sigma, mu, direction))
    return np.asarray(params, dtype=float)


def generate_trajectories(fom, t, n_traj, seed, ranges):
    """FOM trajectories + exact velocity snapshots for ``n_traj`` sampled ICs.

    Returns (X, dX, params) with X, dX of shape (n_traj, 2N, len(t)) and
    params of shape (n_traj, 4). dX = J grad H holds analytic velocities (for
    variationally-consistent OpInf), not finite differences.
    """
    rng        = np.random.default_rng(seed)
    params     = draw_params(n_traj, rng, ranges)
    periodic_L = fom.L if ranges.get("periodic", False) else None
    snaps, derivs = [], []
    for amp, sigma, mu, direction in params:
        z0   = gaussian_ic(fom, amp=amp, width=sigma, x0=mu,
                            direction=direction, periodic_L=periodic_L)
        traj = fom.integrate(z0, t)
        snaps.append(traj)
        derivs.append(fom.rhs_matrix(traj))      # exact dz/dt snapshots
    return np.stack(snaps, axis=0), np.stack(derivs, axis=0), params
