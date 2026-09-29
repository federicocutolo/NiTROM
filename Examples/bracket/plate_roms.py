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
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def canonical_J(dim):
    """Block symplectic matrix J = [[0, I], [-I, 0]] of even size ``dim``."""
    half = dim // 2
    return np.block([[np.zeros((half, half)),  np.eye(half)],
                     [-np.eye(half), np.zeros((half, half))]])


def fom_frequency_band(K, M):
    """Modal angular-frequency band ``(w_min, w_max)`` of the conservative FOM.

    The full operator ``S = J @ blkdiag(K, M^-1)`` has eigenvalues ``+/- i*w``,
    where ``w`` solves the generalized eigenproblem ``K phi = w^2 M phi``. Both
    extremes are found with sparse symmetric ARPACK (shift-invert factorizes the
    sparse ``K`` for the smallest), so the dense 2N x 2N spectrum of ``S`` is
    never formed. ``S`` is infinitesimally symplectic, so ``Re(eig(S)) = 0``.
    """
    from scipy.sparse.linalg import eigsh
    w2_max = eigsh(K, k=1, M=M, which="LM", return_eigenvectors=False)
    try:
        w2_min = eigsh(K, k=1, M=M, sigma=0.0, which="LM", return_eigenvectors=False)
    except Exception:                       # singular K (rigid-body modes) -> w_min ~ 0
        w2_min = eigsh(K, k=6, M=M, which="SM", return_eigenvectors=False)
    return float(np.sqrt(np.abs(w2_min).min())), float(np.sqrt(np.abs(w2_max).max()))


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
    norm_q = np.linalg.norm(Q, "fro")
    norm_p = np.linalg.norm(P, "fro")
    alpha = norm_q / norm_p

    U, s, _ = np.linalg.svd(np.hstack([Q, alpha * P]), full_matrices=False)
    if rank is None:
        explained = np.cumsum(s ** 2) / np.sum(s ** 2)
        rank = int(np.argmax(explained > energy)) + 1
        if rank % 2:  # even rank for hamiltonian ROMs!!
            rank += 1
    Phi_qp = U[:, :rank]
    zero   = np.zeros_like(Phi_qp)
    Phi_cl = np.block([[Phi_qp, zero], [zero, Phi_qp]])
    return Phi_cl, s, rank

def pod_basis(X, N, rank=None, energy=0.99):
    """POD basis from full-state snapshots."""
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if rank is None:
        explained = np.cumsum(s ** 2) / np.sum(s ** 2)
        rank = int(np.argmax(explained > energy)) + 1
    if rank % 2:  # even rank for hamiltonian ROMs!!
        rank += 1
    rank = min(rank, U.shape[1] - (U.shape[1] % 2))  # never exceed snapshot columns
    Phi_pod = U[:, :rank]
    return Phi_pod, s, rank


def _reduced_symplectic(Phi, J):
    """Reduced symplectic structure Jhat = Phi.T J Phi and its inverse."""
    Jhat = Phi.T @ (J @ Phi)              # parenthesized: sparse-J safe (plate)
    return np.asarray(Jhat), np.linalg.inv(np.asarray(Jhat))


# ── VC-H-OpInf (Gruber & Tezaur, SIADS 2024, eq. 9) ──────────────────────────
def vc_h_opinf_operator(Phi, J, trajectories, derivatives):
    """
    Symmetric reduced Hamiltonian operator A_bar, recovered from the Lyapunov
    system  G A_bar + A_bar G = M + M.T  with  Z = Phi.T X,  G = Z Z.T,
    M = Phi.T J.T Xdot Z.T.  Uses EXACT velocity snapshots Xdot (= J W X), not
    finite differences; H is quadratic so only A_bar is inferred.
    """
    r = Phi.shape[1]
    X       = np.concatenate(trajectories, axis=1)
    Xdot    = np.concatenate(derivatives,  axis=1)
    Z = Phi.T @ X
    M = Phi.T @ np.asarray(J.T @ Xdot) @ Z.T      # parenthesized: sparse-J safe (plate)
    G = Z @ Z.T

    # ── Original: full Lyapunov solve with scale-aware Tikhonov regularization ──
    LHS = np.kron(np.eye(r), G) + np.kron(G, np.eye(r))
    eps = 1e-4 * np.trace(G) / r          # scale-aware
    # LHS += eps * np.eye(r * r)
    RHS = M + M.T
    A_bar = np.linalg.solve(LHS, RHS.flatten(order="F")).reshape(r, r, order="F")


    # P          = csc_matrix(np.kron(np.eye(r), G) 
    #                             + np.kron(G, np.eye(r)))
    #     # reg        = 2 * eps * identity(r**2)
    # P += eps * np.eye(r * r)
    # A_bar      = spsolve(P, (RHS).flatten(order='F')).reshape((r,r), order='F')

    # ── Alternative: solve on the range of G (drop its small singular values) ──
    # G = V diag(g) V.T is symmetric PSD, so the equation decouples in its
    # eigenbasis as (g_i + g_j) A~_ij = C~_ij (C = M + M.T). Restricting to the
    # dominant eigenspace drops ill-conditioned directions instead of damping
    # them. Comment out the block above and uncomment this one to try it.
    # rcond = 1e-10
    # g, V = np.linalg.eigh(G)              # ascending, V orthonormal
    # keep = g > rcond * g[-1]
    # g_r, V_r = g[keep], V[:, keep]
    # C_tilde = V_r.T @ (M + M.T) @ V_r
    # A_bar = V_r @ (C_tilde / (g_r[:, None] + g_r[None, :])) @ V_r.T
    # print(f"G range: kept {keep.sum()}/{len(g)} modes (rcond={rcond:.1e})")

    print("-"*50, "Diagnostics for VC-H-OpInf operator A_bar:", "-"*50)
    print("cond(G)     =", np.linalg.cond(G))
    print("cond(Jhat)  =", np.linalg.cond(_reduced_symplectic(Phi, J)[0]))
    print("A_bar sym err =", np.linalg.norm(A_bar - A_bar.T)/np.linalg.norm(A_bar))
    # skewness of dynamics wrt Jhat metric:
    Jhat, Jhat_inv = _reduced_symplectic(Phi, J)
    S = Jhat_inv.T @ A_bar
    print("Jhat S sym check =", np.linalg.norm(Jhat@S + (Jhat@S).T)/np.linalg.norm(Jhat@S))
    print("-"*50, "-"*50)
    print(f"Is Jhat^-T skew? {np.linalg.norm(Jhat_inv.T + Jhat_inv)/np.linalg.norm(Jhat_inv.T):.3e}")
    return 0.5 * (A_bar + A_bar.T)


def vc_h_operator(Phi, J, trajectories, derivatives, rcond=1e-4, spd=True, spd_floor=1e-6,
                  verbose=True):
    """
    VC-H-OpInf symmetric operator A_bar, solved on the RANGE of the data.

    Same Lyapunov system as ``vc_h_opinf_operator``:
        G A_bar + A_bar G = M + M.T ,   Z = Phi.T X ,  G = Z Z.T ,
        M = Phi.T J.T Xdot Z.T
    but robust to rank-deficient snapshots. When X (hence Z) is not full rank --
    few/aligned trajectories -- G = Z Z.T is singular, the Kronecker LHS
    (I⊗G + G⊗I) has zero eigenvalues (g_i + g_j = 0), and the full solve is
    ill-posed. Instead of Tikhonov-damping every direction, we diagonalize the
    SPD Gram G = V diag(g) V.T and solve ONLY on its numerical range: in the
    eigenbasis the Lyapunov equation decouples as

        (g_i + g_j) A~_ij = C~_ij ,   C~ = V_r.T (M + M.T) V_r ,

    so A~_ij = C~_ij / (g_i + g_j) over the kept modes (g_i > rcond * g_max), and
    A_bar = V_r A~ V_r.T. Directions in the null space of G are undetermined by
    the data and are left at zero -- no spurious content, no damping bias.

    Parameters
    ----------
    rcond : relative eigenvalue floor. Modes with g_i <= rcond * g_max are the
            data's null space and are dropped; keep-count = numerical rank of Z.

    Returns the symmetric A_bar (r x r), supported on range(G) only.
    """
    r    = Phi.shape[1]
    X    = np.concatenate(trajectories, axis=1)
    Xdot = np.concatenate(derivatives,  axis=1)
    Z = Phi.T @ X
    M = Phi.T @ np.asarray(J.T @ Xdot) @ Z.T      # parenthesized: sparse-J safe (plate)
    C = M + M.T                                    # symmetric RHS
    G = Z @ Z.T                                    # SPD Gram (rank-deficient here)

    # Diagonalize the SPD Gram and keep its numerical range.
    g, V = np.linalg.eigh(G)                        # ascending eigenvalues, V orthonormal
    keep = g > rcond * g[-1]
    g_r, V_r = g[keep], V[:, keep]                  # (k,), (r, k)

    # Decoupled Lyapunov solve on range(G): A~_ij = C~_ij / (g_i + g_j) (g > 0).
    C_tilde = V_r.T @ C @ V_r
    A_tilde = C_tilde / (g_r[:, None] + g_r[None, :])
    A_bar   = V_r @ A_tilde @ V_r.T                 # zero outside range(G)

    n_keep = int(keep.sum())
    n_trunc = r - n_keep
    if verbose:
        print(f"vc_h_operator: truncated {n_trunc}/{r} Gram modes "
              f"({100 * n_trunc / r:.1f}%), kept {n_keep} (data rank) "
              f"[rcond={rcond:.1e}, cond(G|range)={g_r[-1] / g_r[0]:.2e}]")
    return 0.5 * (A_bar + A_bar.T)                  # re-symmetrize (roundoff)


def opinf_dynamics(Phi, J, A_bar):
    """Reduced dynamics  dz/dt = A z  for VC-H-OpInf."""
    Jhat, Jhat_inv = _reduced_symplectic(Phi, J)
    return Jhat_inv.T @ A_bar


def opinf_init_A2(Phi, J, A_bar):
    """
    OpInf operator in NiTROM's A2 parameterization (warm start). OpInf gives
    z_dot = Jhat^{-T} A_bar z; NiTROM uses A2, related by A2 = 1/2 Jhat^{-T} A_bar Jhat^{-1}.
    Added regularization effect to A_bar to ensure that the reduced dynamics is stable (Re(eig(A_dyn)) <= 0).
    """
    Jhat, Jhat_inv = _reduced_symplectic(Phi, J)
    # A_dyn = opinf_dynamics(Phi, J, A_bar)
    # D, _ = np.linalg.eig(A_dyn)
    # eps = 1.01 * np.max(np.real(D))
    # eps = eps if eps > 0 else 0.0
    # A_dyn_reg -= eps * np.eye(A_dyn.shape[0])  # roundoff symmetrization
    # A_bar_reg = Jhat.T @ A_dyn_reg
    return 0.5 * Jhat_inv.T @ A_bar @ Jhat_inv


# ── Petrov-Galerkin ──────────────────────────────────────────────────────────
def pg_dynamics(Phi, J, W_fom):
    """Reduced dynamics  dz/dt = A z  for Petrov-Galerkin."""
    Jhat, Jhat_inv = _reduced_symplectic(Phi, J)
    E = (J @ Phi).T
    D = Phi @ np.linalg.inv(E @ Phi)
    W_hat = 0.5 * E @ J @ (W_fom + W_fom.T) @ D         # parenthesized
    return W_hat
    

def pg_init_A2(Phi, J, W_fom):
    # NiTROM's dynamics is A_dyn = Jhat (A2 + A2^T) = 2 Jhat A2 for symmetric A2
    # (classes.py), so matching the PG dynamics W_hat requires
    #     Jhat (A2 + A2^T) = W_hat  ==>  A2 = 1/2 Jhat^{-1} W_hat.
    # The 1/2 is essential: without it the init dynamics is 2*W_hat, i.e. the ROM
    # starts at DOUBLE the physical frequency (mirrors the 1/2 in opinf_init_A2).
    # Only the symmetric part of A2 is identifiable (the skew part does not affect
    # the dynamics), and A2 = 1/2 Jhat^{-1} W_hat is symmetric by construction.
    W_hat = pg_dynamics(Phi, J, W_fom)
    _, Jhat_inv = _reduced_symplectic(Phi, J)
    A2 = 0.5 * Jhat_inv @ W_hat
    return A2

