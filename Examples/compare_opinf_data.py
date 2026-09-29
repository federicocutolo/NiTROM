"""
compare_opinf_data.py
=====================
Side-by-side check of the cotangent-lift SVD basis and the VC-H-OpInf
operator inference on the WAVE (Dirichlet) and BRACKET examples.

Both datasets are pushed through *each example's own* functions
(``wave_roms`` / ``plate_roms``) via one identical diagnostic routine, so any
difference in how the snapshot data is treated shows up as a difference in the
printed numbers -- not in the code path.

Run:  python Examples/compare_opinf_data.py
"""
import os
import sys
import glob
import numpy as np
from scipy.sparse import load_npz, issparse

HERE  = os.path.dirname(os.path.abspath(__file__))
WAVE  = os.path.join(HERE, "wave_equation")
BRACK = os.path.join(HERE, "bracket")
for _p in (WAVE, BRACK):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wave_roms                      # noqa: E402  (dense J, multi-trajectory)
import plate_roms                     # noqa: E402  (sparse J, single trajectory)
import plate_cases                    # noqa: E402  (sparse canonical_J for bracket)


def analyze(name, roms, X_list, Xdot_list, W, J):
    """Run one example's SVD + OpInf functions and print the same diagnostics."""
    X    = np.concatenate(X_list,    axis=1)
    Xdot = np.concatenate(Xdot_list, axis=1)
    N    = X.shape[0] // 2

    print(f"\n================  {name}  ================")
    print(f"  roms module        : {roms.__name__}")
    print(f"  state dim 2N       : {X.shape[0]}")
    print(f"  # trajectories     : {len(X_list)}")
    print(f"  # snapshots total  : {X.shape[1]}")
    print(f"  W / J storage      : {'sparse' if issparse(W) else 'dense'} / "
          f"{'sparse' if issparse(J) else 'dense'}")

    # 1. are the derivatives the EXACT velocity Xdot = J W X, column-paired?
    JWX = np.asarray(J @ np.asarray(W @ X))
    rel = np.linalg.norm(Xdot - JWX) / np.linalg.norm(Xdot)
    print(f"  ||Xdot - J W X||/||Xdot||       : {rel:.2e}   (0 => exact-velocity, aligned)")

    # 2. cotangent-lift SVD basis (each example's own routine)
    Phi, s, rank = roms.cotangent_lift_basis(X, N, rank=None, energy=0.99)
    r = Phi.shape[1]
    captured = np.sum(s[:rank] ** 2) / np.sum(s ** 2)
    print(f"  auto rank/field    : {rank}   -> reduced dim r = {r}")
    print(f"  captured [Q|P] energy           : {captured * 100:.4f} %   (target 99%)")
    print(f"  singular decay s0 / s[rank-1]   : {s[0] / s[rank - 1]:.2e}")

    # 3. basis structure (orthonormal + symplectic?)
    Jhat = np.asarray(Phi.T @ np.asarray(J @ Phi))
    orth = np.linalg.norm(Phi.T @ Phi - np.eye(r))
    symp = np.linalg.norm(np.asarray(J @ Phi) - Phi @ Jhat)
    print(f"  ||Phi^T Phi - I||  (orthonormal): {orth:.1e}")
    print(f"  ||J Phi - Phi Jhat|| (symplectic): {symp:.1e}")

    # 4. reduced Gram matrix G = Z Z^T  (the conditioning that drives stability)
    Z  = Phi.T @ X
    gv = np.sort(np.linalg.eigvalsh(Z @ Z.T))[::-1]
    Grank = int((gv > 1e-10 * gv.max()).sum())
    print(f"  G = Z Z^T rank     : {Grank}/{r}   (deficient by {r - Grank})")
    print(f"  cond(G) on excited subspace     : {gv.max() / gv[Grank - 1]:.2e}")

    # 5. operator inference
    A_bar = roms.vc_h_opinf_operator(Phi, J, X_list, Xdot_list)
    A_dyn = roms.opinf_dynamics(Phi, J, A_bar)
    ev_A  = np.linalg.eigvalsh(A_bar)
    ev_d  = np.linalg.eigvals(A_dyn)
    print(f"  A_bar symmetric ||A - A^T||     : {np.linalg.norm(A_bar - A_bar.T):.1e}")
    print(f"  A_bar eig range    : [{ev_A.min():+.2e}, {ev_A.max():+.2e}]   "
          f"{'SPD (ok)' if ev_A.min() > 0 else 'INDEFINITE'}")
    print(f"  A_dyn max Re(lambda): {ev_d.real.max():+.2e}   "
          f"{'STABLE' if ev_d.real.max() < 1.0 else 'UNSTABLE'}")


# ---- WAVE (Dirichlet): many trajectories, dense W & J -----------------------
wdir = os.path.join(WAVE, "Dirichlet", "trajectories")
wX   = [np.load(f) for f in sorted(glob.glob(os.path.join(wdir, "traj_*.npy")))]
wXd  = [np.load(f) for f in sorted(glob.glob(os.path.join(wdir, "deriv_*.npy")))]
wW   = np.load(os.path.join(wdir, "W_fom.npy"))
wJ   = wave_roms.canonical_J(wX[0].shape[0])
analyze("WAVE (Dirichlet)", wave_roms, wX, wXd, wW, wJ)

# ---- BRACKET: single trajectory, sparse W & J ------------------------------
bdir = os.path.join(BRACK, "Bracket", "trajectories")
bX   = [np.load(os.path.join(bdir, "x.npy"))]
bXd  = [np.load(os.path.join(bdir, "xdot.npy"))]
bW   = load_npz(os.path.join(bdir, "A_fom.npz"))
bJ   = plate_cases.canonical_J(bX[0].shape[0])
analyze("BRACKET", plate_roms, bX, bXd, bW, bJ)

print("\n(Identical diagnostic routine for both -> any difference above is in the DATA, not the code.)")
