"""
Prepare the NiTROM inputs for the 3D linear-elastic bracket, straight from the
raw FOM artifacts (Exodus snapshots + FE mass/stiffness matrices). This unifies
the former open_exodus_file.py (Exodus -> state matrices) and prepare_data.py
(state matrices -> NiTROM layout) into a single step.

Reads (from bracket_for_Fede/):
    bracket_velIC_100.e   Exodus FOM output: disp / solution_dot / solution_dotdot
    mass.mm               consistent mass matrix      M   (interleaved x,y,z DOFs)
    stiff.mm              stiffness matrix            K   (same DOF ordering)

Writes (into Bracket/trajectories/):
    x.npy       state matrix      [q; p],  p = M q_dot           (2*3N, Nt)
    xdot.npy    velocities        [q_dot; p_dot], p_dot = M q_ddot
    time.npy    snapshot times                                  (Nt,)
    weight.npy  scalar energy weight
    Phi_cl.npy  cotangent-lift symplectic basis                 (2*3N, 2*RANK)
    A_fom.npz   FOM Hamiltonian operator  blkdiag(K, M^-1)      (H = 1/2 x^T A x)
    fom_freqs.npy  FOM spectral summary [re_max, w_min, w_max] of S=J·A (rad/s)

The canonical-Hamiltonian state is x = [q; p] with q the (interleaved) nodal
displacement and p = M q_dot the momentum; the Hamiltonian is H = 1/2 x^T A x
with A = blkdiag(K, M^-1). Velocities are the exact FOM rates [q_dot; p_dot],
not finite differences. This mirrors the reference bracket_script_new.py.

NOTE: the mass matrix is consistent (not lumped), so M^-1 is dense; A_fom is
therefore a densely-populated sparse matrix (~3N x 3N nonzeros).

Usage:
    python prepare_data.py [path/to/bracket_velIC_100.e]
"""
import os
import sys

import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix, eye, bmat, block_diag, save_npz
from scipy.sparse.linalg import spsolve, eigs
import exodusii
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import plate_roms
from plate_cases import get_case

# ── Config ────────────────────────────────────────────────────────────────────
TESTCASE = "bracket"          # which testcase to prepare: "bracket" | "cantilever_plate"
_case = get_case(TESTCASE)
EXO   = sys.argv[1] if len(sys.argv) > 1 else _case.exo
MASS  = _case.mass
STIFF = _case.stiff

# Optional structure-preserving pre-compression of the FOM data. When on, the
# state/velocity/A_fom are projected onto a shared POD basis (symplectically, so
# the canonical [q;p] structure is preserved); when off, the full FOM is used.
COMPRESS        = False        # symplectic data pre-compression on/off
TRAIN_FRAC      = 0.60         # fraction of snapshots used for training (rest = held-out test)


def interleaved(exo, base, N, Nt):
    """Stack node variable ``base``_{x,y,z} over all time steps -> (3N, Nt),
    interleaved as [x0,y0,z0, x1,y1,z1, ...] to match the M / K DOF ordering."""
    out = np.zeros((3 * N, Nt))
    for c, comp in enumerate("xyz"):
        for t in range(Nt):
            out[c::3, t] = exo.get_node_variable_values(f"{base}_{comp}", t + 1)
    return out


def assemble_state(exo, mass):
    """Build (x, xDot) with x = [q; p], p = M q_dot, from the Exodus snapshots.
    Mirrors the reference assemble_FOM (momentum = mass @ velocity)."""
    N, Nt = exo.num_nodes(), exo.num_times()
    q     = interleaved(exo, "disp",            N, Nt)
    qDot  = interleaved(exo, "solution_dot",    N, Nt)
    qDDot = interleaved(exo, "solution_dotdot", N, Nt)
    p     = mass @ qDot
    pDot  = mass @ qDDot
    x_     = np.concatenate([q, p],       axis=0)      # (2*3N, Nt)
    xDot_  = np.concatenate([qDot, pDot], axis=0)

    print(f"  Assembled state x {x_.shape}, xDot {xDot_.shape} from Exodus snapshots")
    return x_, xDot_


def symplectic_compression(x_, xDot_, energy=0.99):
    """Structure-preserving pre-compression of the FOM data.

    Uses ONE shared POD basis U_c for both the q and p halves, so the
    compression map T = blkdiag(U_c, U_c) is symplectic: the canonical [q; p]
    block structure is preserved (canonical_J stays canonical and the
    downstream cotangent lift stays a genuine reduction). The velocity is
    compressed with the SAME map so the exact rate relation Xdot = J A X holds
    in the compressed coordinates.

    Returns (x, xDot, U_c) with x, xDot of shape (2*rc, Nt) and U_c of (3N, rc).
    """
    Nf = x_.shape[0] // 2                                  # = 3N
    snap = np.hstack([x_[:Nf], x_[Nf:]])                   # [q | p] snapshots (3N, 2Nt)
    Uc, s, _ = np.linalg.svd(snap, full_matrices=False)
    rc = int(np.argmax(np.cumsum(s ** 2) / np.sum(s ** 2) > energy)) + 1
    Uc = Uc[:, :rc]                                        # (3N, rc)

    x    = np.vstack([Uc.T @ x_[:Nf],    Uc.T @ x_[Nf:]])      # (2*rc, Nt)
    xDot = np.vstack([Uc.T @ xDot_[:Nf], Uc.T @ xDot_[Nf:]])  # (2*rc, Nt)

    captured = np.sum(s[:rc] ** 2) / np.sum(s ** 2)
    print(f"  Symplectic compression: rc {rc}/field -> state dim {2 * rc} "
          f"({captured * 100:.4f}% energy)")
    return x, xDot, Uc, s


def plot_compression_svd(s, out_dir):
    """Explained-variance plot of the DATA-COMPRESSION SVD (full-space [q | p]
    snapshots -> Uc), in the same style as the basis svd plot below. This is
    the stage that sets the compressed state dim 2*rc; the svd.png plot further
    down shows the second-stage basis SVD *inside* the compressed space."""
    explained_var = np.cumsum(s ** 2) / np.sum(s ** 2)
    ranks = np.arange(1, len(s) + 1)
    rc = int(np.argmax(explained_var > 0.99)) + 1
    if rc % 2:
        rc += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ranks, explained_var * 100, marker="o", color="k")
    ax.set_xlabel(r"Index $i$", fontsize=14)
    ax.set_ylabel(r"Captured POD energy [%]", fontsize=14)
    ax.set_ylim(20, 105)
    plt.axvline(rc, color="k", linestyle="--", label=f"99% energy at rank={rc}")
    ax.set_title(r"Data compression SVD (full-space $\mathbf{x}$ $\in \mathbb{R}^{2N}$)", fontsize=14)
    ax.ticklabel_format(style="plain", axis="both")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "svd_compression.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "svd_compression.pdf"), bbox_inches="tight")
    plt.close(fig)


def build_A_fom(mass, stiff):
    """FOM Hamiltonian operator A = blkdiag(K, M^-1), so H = 1/2 x^T A x.

    M is a consistent FE mass matrix, so M^-1 is dense; A is returned as a
    (densely-populated) sparse csc whose only use downstream is matmul (A @ X,
    A @ Phi), exactly like the reference bracket_script_new.py.
    """
    n    = mass.shape[0]                                   # 3N
    Minv = spsolve(mass.tocsc(), eye(n, format="csc"))    # dense inverse
    Minv.eliminate_zeros()  # drop explicit zeros from the dense inverse
    A = bmat([[stiff, None], [None, Minv]], format="csc")
    A.eliminate_zeros()  # drop explicit zeros from the dense inverse
    return A


def hamiltonian(A, X):
    """H(t_i) = 1/2 x_i^T A x_i for each column of X (works for sparse A)."""
    return 0.5 * np.einsum("ij,ij->j", np.asarray(A @ X), np.asarray(X))


def fom_spectrum_S(A, k=6, which="LM"):
    """Assemble S = J @ A (J canonical) and return k of its eigenvalues, sparsely.

    S is the 2N x 2N Hamiltonian dynamics matrix (dz/dt = S z, S = J A). It is a
    Hamiltonian matrix, so for symmetric-SPD A the spectrum is purely imaginary,
    +/- i*w. ARPACK (scipy.sparse.linalg.eigs) is asked for the k *largest-
    magnitude* eigenvalues -> the highest-frequency modes, which is robust here.
    Do NOT search by real part (which='LR'): on a purely-imaginary spectrum that
    is what returns spurious real parts (~1e6). Returns the complex eigenvalues.
    """
    n = A.shape[0] // 2
    I = eye(n, format="csc")
    J = bmat([[None, I], [-I, None]], format="csc")        # [[0, I], [-I, 0]]
    S = (J @ A.tocsc()).tocsc()
    return eigs(S, k=k, which=which, return_eigenvectors=False)


def main():
    case    = _case
    out_dir = case.data_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"Preparing NiTROM data for testcase '{case.name}' in {out_dir}")
    # ── FE operators + Hamiltonian operator A = blkdiag(K, M^-1) ──────────────
    M = csc_matrix(mmread(MASS))
    K = csc_matrix(mmread(STIFF))
    A = build_A_fom(M, K)
    

    print("─" * 64)
    print("Operators  (A = blkdiag(K, M^-1),  H = 1/2 x^T A x)")
    print(f"  K   shape {str(K.shape):>16}   nnz {K.nnz:>12,}")
    print(f"  M   shape {str(M.shape):>16}   nnz {M.nnz:>12,}")
    print(f"  A   shape {str(A.shape):>16}   nnz {A.nnz:>12,}")
    print("─" * 64)

    # ── State / velocity matrices from the Exodus snapshots ───────────────────
    exo   = exodusii.ExodusIIFile(EXO, mode="r")
    N, Nt = exo.num_nodes(), exo.num_times()
    times = np.asarray(exo.get_times())
    print(f"Loaded {EXO}")
    print(f"  nodes {N}  ->  state dim 2*3N = {2 * 3 * N},  {Nt} snapshots "
          f"in [0, {times[-1]:g}]")

    if Nt < 2:
        raise ValueError(
            f"{EXO} reports only {Nt} time step(s); the ROM pipeline needs a "
            f"transient time-series (Nt >= 2) to build a basis and infer "
            f"dynamics. Verify this Exodus file is the full transient output, "
            f"not a single initial-condition snapshot (e.g. `ncdump -h {EXO} | "
            f"grep time_step`)."
        )

    xData_, xDot_ = assemble_state(exo, M)

    if COMPRESS:
        xData, xDot, Uc, s_comp = symplectic_compression(xData_, xDot_, energy=0.99)
        plot_compression_svd(s_comp, out_dir)
    else:
        xData, xDot, Uc = xData_, xDot_, None             # full FOM, no reduction
    n2 = xData.shape[0]
    N_half = n2 // 2                                       # = 3N (or rc if compressed)

    # ── Train/test split over snapshots (single trajectory) ───────────────────
    # The basis and all operators are built from the first TRAIN_FRAC snapshots;
    # the remaining snapshots are a held-out (in-time) test. The full trajectory
    # is still saved for evaluation; a split index marks train vs test times.
    n_train = int(round(TRAIN_FRAC * xData.shape[1]))
    xData_tr = xData[:, :n_train]                          # train snapshots (for the basis)
    print(f"Train/test snapshot split: {n_train}/{xData.shape[1]} "
          f"({100 * TRAIN_FRAC:.0f}% train, t_split = {times[n_train]:.4g})")

    # ── Tip node = center of the free-end face (for displacement plots) ────────
    coords = np.asarray(exo.get_coords())                 # (N, 3) in node order (= DOF order)
    ext    = coords.max(0) - coords.min(0)
    axis   = int(np.argmax(ext))                          # long (beam) axis
    lo, hi = coords[:, axis].min(), coords[:, axis].max()
    cen    = coords[:, axis].mean()
    end    = lo if abs(lo - cen) > abs(hi - cen) else hi  # more-protruding end = tip
    face   = np.where(np.abs(coords[:, axis] - end) <= 0.02 * ext[axis])[0]
    other  = [a for a in range(3) if a != axis]
    fc     = coords[np.ix_(face, other)].mean(0)          # cross-section center of the tip face
    tip    = int(face[np.argmin(np.linalg.norm(coords[np.ix_(face, other)] - fc, axis=1))])
    print(f"Tip node {tip} at {np.round(coords[tip], 4)}  "
          f"(long axis '{'xyz'[axis]}', {len(face)} face nodes; DOFs {3*tip}:{3*tip+3})")

    # ── FOM vs compressed data sizes ──────────────────────────────────────────
    dim_fom, dim_cmp = xData_.shape[0], xData.shape[0]     # state dims (2*3N vs 2*rc)
    mb = lambda a: a.nbytes / 1e6
    print("─" * 64)
    print(f"Data sizes  (COMPRESS={COMPRESS})")
    print(f"  {'':10} {'state dim':>12} {'x shape':>18} {'size':>12}")
    print(f"  {'FOM':10} {dim_fom:>12,} {str(xData_.shape):>18} {mb(xData_):>9.1f} MB")
    print(f"  {'compressed':10} {dim_cmp:>12,} {str(xData.shape):>18} {mb(xData):>9.1f} MB")
    if COMPRESS:
        print(f"  reduction  {dim_fom / dim_cmp:>11.1f}x  "
              f"({dim_cmp}/{dim_fom} states, {100 * mb(xData) / mb(xData_):.1f}% of FOM size)")
    else:
        print("  (no compression: compressed == FOM)")
    print("─" * 64)
    print(f"  IC ||q0||={np.linalg.norm(xData[:N_half, 0]):.3e}  "
          f"||p0||={np.linalg.norm(xData[N_half:, 0]):.3e}")

    # ── FOM spectrum of S = J A ───────────────────────────────────────────────
    # S is a Hamiltonian matrix (J S = -A is symmetric), so for symmetric SPD A
    # its eigenvalues are purely imaginary: Re(eig(S)) = 0 *exactly*. We do NOT
    # eig S for "max Re": it is 2N x 2N (dense eig infeasible), and ARPACK's
    # rightmost-eigenvalue search returns spurious real parts (~1e6) when the
    # whole spectrum sits on the imaginary axis. The trustworthy, cheap stand-in
    # is the asymmetry of A -- while ||A - A^T|| ~ 0 the FOM is conservative; a
    # damped/non-conservative FOM would surface here as a nonzero residual.
    # The frequencies |Im| = w come from the generalized problem K phi = w^2 M
    # phi (sparse eigsh, ~0.2 s). Saved so the train/test scripts (which hold
    # only A_fom, not M) can report the band too.
    asym   = np.sqrt((A - A.T).multiply(A - A.T).sum() / A.multiply(A).sum())
    w_min, w_max = plate_roms.fom_frequency_band(K, M)
    np.save(os.path.join(out_dir, "fom_freqs.npy"), np.array([w_min, w_max]))

    # Direct sparse eig of the assembled S = J A (cross-check the band above).
    # |Im| of the largest-magnitude eigenvalues should match w_max; |Re| should
    # be ~0 for a conservative FOM (any drift here flags spurious ARPACK output).
    ev_S = fom_spectrum_S(A, k=6, which="LM")

    dt = times[1] - times[0]   # data sampling interval
    print("─" * 64)
    print("FOM spectrum  S = J·A")
    print(f"  Im(eig)   |Im| in [{w_min:.3e}, {w_max:.3e}] rad/s   (eigenfrequencies)")
    print(f"  eig(S)    |Im| max {np.abs(ev_S.imag).max():.3e} rad/s "
          f"(vs w_max {w_max:.3e}),  |Re| max {np.abs(ev_S.real).max():.1e}")
    print(f"  Re(eig)   ||A - A^T|| / ||A|| = {asym:.3e}   (0 => conservative)")
    print(f"  cond(A)   = {(w_max / w_min) ** 2:.3e}")
    print(f"  Nyquist   f_max {w_max / (2 * np.pi):.3e} Hz  vs  f_sample {1.0 / dt:.3e} Hz"
          f"   -> {'OK' if w_max / (2 * np.pi) < 1.0 / dt else 'VIOLATED'}")
    print(f"  Resolved  f_min {w_min / (2 * np.pi):.3e} Hz  vs  1/T {1.0 / (times[-1] - times[0]):.3e} Hz"
            f"   -> {'OK' if w_min / (2 * np.pi) > 1.0 / (times[-1] - times[0]) else 'VIOLATED'}")
    print("─" * 64)


    # ── ROM bases -> BOTH Phi_cl.npy and Phi_pod.npy ──────────────────────────
    # Built from TRAIN snapshots only, so the test snapshots are genuinely unseen;
    # train_nitrom / test_nitrom select one via their BASIS switch.
    Phi_cl, s_cl, rank = plate_roms.cotangent_lift_basis(xData_tr, N_half, rank=None, energy=0.99)
    captured = np.sum(s_cl[:rank] ** 2) / np.sum(s_cl ** 2)
    print(f"  cotangent-lift basis: rank {rank}/field -> Phi_cl {Phi_cl.shape} "
          f"({captured * 100:.4f}% energy)")

    # Plain POD of the full [q; p] state, truncated INDEPENDENTLY of the CL
    # basis: same 99%-energy criterion applied to its OWN SVD (pod_basis keeps
    # r even, since an odd r makes Jhat = Phi^T J Phi exactly singular).
    Phi_pod, s_pod, r_pod = plate_roms.pod_basis(xData_tr, N_half, rank=None, energy=0.99)
    captured_pod = np.sum(s_pod[:r_pod] ** 2) / np.sum(s_pod ** 2)
    print(f"  plain POD basis:      rank {r_pod}      -> Phi_pod {Phi_pod.shape} "
          f"({captured_pod * 100:.4f}% energy)")

    # ── CL symplectic-projection compression error on the EXODUS data ─────────
    # P = U ((JU)^T U)^{-1} (JU)^T  (oblique symplectic projector, U = Phi_cl):
    # reports ||X - P X||_F on the raw FE snapshots X = [q; M q_dot]. Only
    # meaningful when the basis lives in the FULL space (COMPRESS = False).
    if Uc is None:
        I_h  = eye(N_half, format="csc")
        J_cl = bmat([[None, I_h], [-I_h, None]], format="csc")
        JU   = np.asarray(J_cl @ Phi_cl)
        PX   = Phi_cl @ np.linalg.solve(JU.T @ Phi_cl, JU.T @ xData_)
        err_abs = np.linalg.norm(xData_ - PX)
        err_rel = err_abs / np.linalg.norm(xData_)
        print(f"  CL symplectic projection error on Exodus data: "
              f"||X - U((JU)^T U)^-1 (JU)^T X||_F = {err_abs:.6e}  "
              f"(relative {err_rel:.3e})")

    # ── SVD plot: explained variance of BOTH bases (black & white) ────────────
    # CL curve: singular values of the stacked [Q | P] block (rank = modes/field);
    # POD curve: singular values of the full [q; p] snapshot matrix (rank = r).
    ev_cl  = np.cumsum(s_cl ** 2)  / np.sum(s_cl ** 2)
    ev_pod = np.cumsum(s_pod ** 2) / np.sum(s_pod ** 2)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(np.arange(1, len(s_cl) + 1),  ev_cl * 100,  color="k", linestyle="-",
            marker="o", markerfacecolor="k",    label=r"Cotangent lift $[Q\,|\,P]$")
    ax.plot(np.arange(1, len(s_pod) + 1), ev_pod * 100, color="k", linestyle="--",
            marker="s", markerfacecolor="none", label="POD (full state)")
    ax.axvline(rank,  color="k", linestyle=":",  linewidth=1,
               label=f"CL rank = {rank}")
    ax.axvline(r_pod, color="k", linestyle="-.", linewidth=1,
               label=f"POD rank = {r_pod}")
    ax.set_xlabel(r"Index $i$", fontsize=14)
    ax.set_ylabel(r"Captured energy [%]", fontsize=14)
    ax.set_ylim(20, 105)
    ax.set_title("Energy captured by bases choice", fontsize=14)
    ax.ticklabel_format(style='plain', axis='both')
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "svd.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "svd.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Per-trajectory energy weight (train snapshots, full FOM space) ─────────
    weight = np.linalg.norm(xData_[:, :n_train]) ** 0.5

    # Full-FOM energy time series H(t) = 1/2 x^T A x on the UNCOMPRESSED state and
    # operator (all modes). Saved as a scalar series so the test script can plot
    # the true FOM energy drift without holding the full A / full trajectory.
    fom_energy = hamiltonian(A, xData_)                    # (Nt,) full-order DATA energy (no integration)

    # When compressing, A must live in the same (2*rc) space as Phi_cl / x, else
    # train_nitrom's A_fom @ Phi mismatches. T = blkdiag(Uc, Uc) is symplectic,
    # so A_save = T^T A T stays exactly blkdiag(K~, M~^-1).
    if Uc is not None:
        T      = block_diag([csc_matrix(Uc), csc_matrix(Uc)], format="csc")  # (2*3N, 2*rc)
        A_save = csc_matrix(T.T @ (A @ T))
        print(f"  A_fom projected to compressed space: {A.shape} -> {A_save.shape}")
    else:
        A_save = A

    # ── Write NiTROM layout ───────────────────────────────────────────────────
    # FULL trajectory (x/xdot/time) is saved for evaluation; the *_train.npy files
    # hold only the first n_train snapshots and are what train_nitrom consumes
    # (its MPI pool splits them into one time segment per rank). n_train marks the
    # train/test boundary for the test-script plots.
    np.save(os.path.join(out_dir, "x.npy"),        xData)
    np.save(os.path.join(out_dir, "xdot.npy"),     xDot)
    np.save(os.path.join(out_dir, "time.npy"),     times)
    np.save(os.path.join(out_dir, "x_train.npy"),    xData[:, :n_train])
    np.save(os.path.join(out_dir, "xdot_train.npy"), xDot[:, :n_train])
    np.save(os.path.join(out_dir, "time_train.npy"), times[:n_train])
    np.save(os.path.join(out_dir, "n_train.npy"),  np.asarray([n_train]))
    np.save(os.path.join(out_dir, "weight.npy"),   np.asarray([weight]))
    np.save(os.path.join(out_dir, "fom_energy.npy"), fom_energy)   # full-order DATA H(t)
    np.save(os.path.join(out_dir, "Phi_cl.npy"),   Phi_cl)
    np.save(os.path.join(out_dir, "Phi_pod.npy"),  Phi_pod)
    save_npz(os.path.join(out_dir, "A_fom.npz"), A_save)

    # Full-space recovery for displacement plots: Uc lifts a compressed field back
    # to the 3N physical DOFs (q_full = Uc @ q_compressed); tip_node + coords name
    # the node to probe. Saved only when compressing (else the state is full).
    if Uc is not None:
        np.save(os.path.join(out_dir, "Uc.npy"), Uc)              # (3N, rc) compression encoder
    np.save(os.path.join(out_dir, "coords.npy"),   coords)        # (N, 3) node coordinates
    np.save(os.path.join(out_dir, "tip_node.npy"), np.asarray([tip]))

    print(f"Wrote NiTROM inputs to {out_dir}  (train {n_train}/{xData.shape[1]} snapshots)")


if __name__ == "__main__":
    main()
