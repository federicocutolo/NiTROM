"""
Baseline reduced-order models for the wave-equation example.

VC-H-OpInf and Petrov-Galerkin are *baseline ROMs*: they are consumed both by
``train_nitrom.py`` (to warm-start the NiTROM optimization) and by
``test_nitrom.py`` (as comparison baselines). They therefore live here, upstream
of both, as the single source of truth for the operators and for the conversion
between the OpInf/PG operators and NiTROM's A2 parameterization.

Each baseline is exposed in two views, because it plays two roles:
  * ``*_init_A2``  -> NiTROM's A2 parameterization, for warm-starting training
  * ``*_dynamics`` -> reduced operator A in  dz/dt = A z, for integration/plots
Same underlying operator, two roles. The OpInf operator A_bar is cheap to
recompute (a single Lyapunov solve), so it is never persisted to disk; both
scripts call these functions instead, which removes the train<->test ordering
dependency that an ``A_bar.npy`` artifact would impose.
"""
import numpy as np


def canonical_J(dim):
    """Block symplectic matrix J = [[0, I], [-I, 0]] of even size ``dim``."""
    half = dim // 2
    return np.block([[np.zeros((half, half)),  np.eye(half)],
                     [-np.eye(half), np.zeros((half, half))]])


def cotangent_lift_basis(X, N, rank=None, energy=0.99):
    """Peng-Mohseni cotangent-lift symplectic POD basis from full-state snapshots.

    POD of the stacked [Q | P] block (Q = X[:N], P = X[N:]) so the reduced
    symplectic form Jhat = Phi.T J Phi is canonical ([[0, I],[-I, 0]], cond = 1)
    by construction. This is the single source for the basis used by the data
    generators (main.py) and the rank sweep.

    rank : explicit spatial rank (POD modes per field). If None, the smallest
           rank capturing ``energy`` of the [Q|P] variance is used.
    Returns (Phi_cl, sing_vals, spatial_rank) with Phi_cl of shape (2N, 2*rank).
    """
    Q, P = X[:N], X[N:]
    U, s, _ = np.linalg.svd(np.hstack([Q, P]), full_matrices=False)
    if rank is None:
        explained = np.cumsum(s ** 2) / np.sum(s ** 2)
        rank = int(np.argmax(explained > energy)) + 1
    Phi_qp = U[:, :rank]
    zero   = np.zeros_like(Phi_qp)
    Phi_cl = np.block([[Phi_qp, zero], [zero, Phi_qp]])
    return Phi_cl, s, rank


def pod_basis(X, rank=None, energy=0.99):
    """Plain POD basis of the full [q; p] snapshot matrix.

    rank : explicit reduced dimension. If None, the smallest rank capturing
           ``energy`` of the variance is used. Kept EVEN either way (an odd r
           makes Jhat = Phi^T J Phi exactly singular) and capped at the
           available snapshot columns (numpy slicing would truncate silently).
    Returns (Phi_pod, sing_vals, rank).
    """
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if rank is None:
        explained = np.cumsum(s ** 2) / np.sum(s ** 2)
        rank = int(np.argmax(explained > energy)) + 1
    if rank % 2:
        rank += 1
    rank = min(rank, U.shape[1] - (U.shape[1] % 2))
    return U[:, :rank], s, rank


def _reduced_symplectic(Phi, J):
    """Reduced symplectic structure Jhat = Phi.T J Phi and its inverse."""
    Jhat = Phi.T @ J @ Phi
    return Jhat, np.linalg.inv(Jhat)


# ── VC-H-OpInf (Gruber & Tezaur, SIADS 2024, eq. 9) ──────────────────────────
def vc_h_opinf_operator(Phi, J, trajectories, derivatives):
    """
    Symmetric reduced Hamiltonian operator A_bar, recovered from the Lyapunov
    system  G A_bar + A_bar G = M + M.T  with  Z = Phi.T X,  G = Z Z.T,
    M = Phi.T J.T Xdot Z.T.  Uses EXACT velocity snapshots Xdot (= J W X), not
    finite differences; H is quadratic so only A_bar is inferred.
    """
    r = Phi.shape[1]
    X    = np.concatenate(trajectories, axis=1)
    Xdot = np.concatenate(derivatives,  axis=1)
    Z = Phi.T @ X
    M = (Phi.T @ J.T) @ Xdot @ Z.T
    G = Z @ Z.T
    LHS = np.kron(np.eye(r), G) + np.kron(G, np.eye(r))
    A_bar = np.linalg.solve(LHS, (M + M.T).flatten(order="F")).reshape(r, r, order="F")
    return 0.5 * (A_bar + A_bar.T)


def opinf_dynamics(Phi, J, A_bar):
    """Reduced dynamics  dz/dt = A z  for VC-H-OpInf."""
    _, Jhat_inv = _reduced_symplectic(Phi, J)
    return Jhat_inv.T @ A_bar


def opinf_init_A2(Phi, J, A_bar):
    """
    OpInf operator in NiTROM's A2 parameterization (warm start). OpInf gives
    z_dot = Jhat^{-T} A_bar z; NiTROM uses A2, related by A2 = 1/2 Jhat^{-T} A_bar Jhat^{-1}.
    """
    _, Jhat_inv = _reduced_symplectic(Phi, J)
    return 0.5 * Jhat_inv.T @ A_bar @ Jhat_inv


# ── Petrov-Galerkin ──────────────────────────────────────────────────────────
def pg_dynamics(Phi, J, W_fom):
    """Reduced dynamics  dz/dt = A z  for Petrov-Galerkin, BASIS-GENERAL.

    A = E (J W) D with encoder E = (J Phi)^T, decoder D = -Phi Jhat^{-1}
    -> A = -W_hat Jhat^{-1}; conserves the projected energy D^T W_fom D for any
    invertible skew Jhat. For a canonical Jhat (cotangent lift,
    Jhat^{-1} = -Jhat) this equals the old shortcut W_hat @ Jhat, which is
    WRONG for a generic (e.g. POD) basis."""
    Jhat, Jhat_inv = _reduced_symplectic(Phi, J)
    W_hat = Phi.T @ W_fom @ Phi
    return -W_hat @ Jhat_inv


def pg_init_A2(Phi, J, W_fom):
    """Petrov-Galerkin operator in NiTROM's A2 parameterization (warm start),
    basis-general: Jhat (A2 + A2^T) = -W_hat Jhat^{-1} = pg_dynamics
    -> A2 = -1/2 Jhat^{-1} W_hat Jhat^{-1} (symmetric)."""
    Jhat, Jhat_inv = _reduced_symplectic(Phi, J)
    W_hat = Phi.T @ W_fom @ Phi
    return -0.5 * Jhat_inv @ W_hat @ Jhat_inv
