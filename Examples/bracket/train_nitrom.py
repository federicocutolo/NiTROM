"""
Train a Hamiltonian NiTROM on the 3D linear-elastic bracket.

Run:
    python  train_nitrom.py                          # serial (one segment)
    mpiexec -n 4 python -u train_nitrom.py           # MPI: split into 4 time segments

The training data is a single trajectory stored as one (states x times) matrix
x.npy, with the exact velocities (Xdot = J A X) in xdot.npy. For parallelization
the trajectory is split into one contiguous time segment per MPI rank; each
segment is integrated as a short trajectory from its own true FOM initial state
(multiple-shooting), and the per-rank costs/gradients are summed across ranks.

Inputs are produced by prepare_data.py (x/xdot/time/weight/Phi_cl/A_fom) in
Bracket/trajectories. Baseline operators (VC-H-OpInf, Petrov-Galerkin) come
from plate_roms, exactly as the wave example uses wave_roms.

The run is labeled by its initialization -- the reduced BASIS ("cl" | "pod")
and the TENSOR seeding A2 ("pg" | "opinf") -- and its solution is written to
the matching container Bracket/output/<basis>_<tensor>/ (e.g. cl_opinf), so
the four initialization cases never overwrite each other. test_nitrom.py then
compares all trained cases for one basis.
"""
import os

# Must run before numpy import: pin BLAS/OpenMP threads per process.
def set_cpu_threads(n):
    n = str(int(n))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = n

set_cpu_threads(1)

import sys
import time as tlib
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

import pymanopt
import pymanopt.manifolds  as manifolds
import pymanopt.optimizers as optimizers
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from NiTROM.Optimization_Functions       import classes, nitrom_functions
from NiTROM.PyManopt_Functions.my_pymanopt_classes import myAdaptiveLineSearcher
from plate_cases import get_case, canonical_J, nitrom_output_dir, ensure_nitrom_dirs
import plate_roms


# ============================================================================
# Helpers
# ============================================================================
def load_train_snapshots(data_dir):
    """Single-trajectory state / exact-velocity matrices (Xdot = J A X) for OpInf.

    Both are stored as one ``(states, times)`` matrix (x.npy / xdot.npy). The
    OpInf warm-start fits a global least-squares over all snapshots, so it uses
    the full undivided trajectory regardless of how training splits it.
    """
    x    = np.load(os.path.join(data_dir, "x_train.npy"))
    xdot = np.load(os.path.join(data_dir, "xdot_train.npy"))
    return [x], [xdot]


def make_opt_obj(which_fix, which_times):
    return classes.optimization_objects(
        pool,
        which_trajs=np.arange(pool.my_n_traj),
        which_times=which_times,
        leggauss_deg=leggauss_deg,
        nsave_rom=nsave_rom,
        poly_comp=poly_comp,
        hamiltonian=True,
        which_fix=which_fix,
        dt_rom=dt_rom,
    )


def init_basis(basis):
    """
    Load the reduced basis Phi (its columns span the ROM subspace).

      "cl"      -> cotangent-lift symplectic basis   (Phi_cl; Jhat canonical)
      "pod"     -> plain POD basis                    (Phi_pod; NOT symplectic)
      "random"  -> random matrix of Phi_cl's shape    (cold start)

    Restarting from a previous NiTROM run is NOT a basis choice: set
    RESTART = True and the (BASIS, TENSOR) pair selects the container whose
    Phi_nit / A2_nit are resumed.
    """
    if basis == "cl":
        Phi = np.load(os.path.join(data_dir, "Phi_cl.npy"))
    elif basis == "pod":
        Phi = np.load(os.path.join(data_dir, "Phi_pod.npy"))
    elif basis == "random":
        Phi_copy = np.load(os.path.join(data_dir, "Phi_cl.npy"))
        n, r = Phi_copy.shape
        Phi  = np.random.rand(n, r)
    else:
        raise ValueError(f"unknown BASIS={basis!r}; use 'cl', 'pod', or 'random'")

    if pool.rank == 0:
        print(f"Basis : {basis:8s} -> Phi {Phi.shape}")
    return Phi


def init_tensor(tensor, Phi):
    """
    Seed the reduced Hamiltonian operator A2 for a *given* basis Phi.

      "pg"      -> Petrov-Galerkin projection of A_fom
      "opinf"   -> VC-H-OpInf (Lyapunov solve on the reduced snapshots)
      "random"  -> small random perturbation (cold start)

    Returns (A2, A_dyn); A_dyn is the reduced dynamics matrix for diagnostics
    (None for the random seed, which has no closed-form dynamics).
    """
    r     = Phi.shape[1]
    A_dyn = None

    if tensor in ("pg", "opinf"):
        J = canonical_J(Phi.shape[0])                           # sparse 2N x 2N
        if tensor == "pg":
            A_fom = load_npz(os.path.join(data_dir, "A_fom.npz"))   # blkdiag(K, M^-1)
            if A_fom.shape[0] != Phi.shape[0]:
                raise ValueError(
                    f"A_fom ({A_fom.shape[0]}) and Phi ({Phi.shape[0]}) live in "
                    f"different spaces. Re-run prepare_data.py so both are built with "
                    f"the same COMPRESS setting (A_fom is projected only when the data "
                    f"is compressed)."
                )
            A2    = plate_roms.pg_init_A2(Phi, J, A_fom)
            A_dyn = plate_roms.pg_dynamics(Phi, J, A_fom)

        else:
            trajs, derivs = load_train_snapshots(data_dir)
            A_bar = plate_roms.vc_h_operator(Phi, J, trajs, derivs,   # range-restricted lyap solve for rank-deficient data
                                             verbose=(pool.rank == 0))
            A_bar = make_definite(A_bar, sign="pos")                  # now SPD -> imaginary spectrum
            if pool.rank == 0:
                w = np.linalg.eigvalsh(A_bar)                         # real eigs (A_bar symmetric), ascending
                print(f"A_bar : {A_bar.shape[0]}x{A_bar.shape[0]}  "
                      f"eig in [{w.min():+.3e}, {w.max():+.3e}]  "
                      f"SPD={w.min() > 0}")
            A2    = plate_roms.opinf_init_A2(Phi, J, A_bar)
            A_dyn = plate_roms.opinf_dynamics(Phi, J, A_bar)

    elif tensor == "random":
        A2 = np.empty((r, r))
        if pool.rank == 0:
            A2[:] = np.random.randn(r, r) * 1e-4
        pool.comm.Bcast(A2, root=0)                             # same seed on all ranks

    else:
        raise ValueError(
            f"unknown TENSOR={tensor!r}; use 'pg', 'opinf', or 'random'")

    if pool.rank == 0:
        print(f"Tensor: {tensor:8s} -> A2 {A2.shape}, asymmetry "
              f"{np.linalg.norm(A2 - A2.T):.3e}")
        if A_dyn is not None:
            eig = np.linalg.eigvals(A_dyn)
            print(f"         max Re(A_dyn) = {eig.real.max():+.3e}   "
                  f"max |Im(A_dyn)| = {np.abs(eig.imag).max():+.3e}")
    return A2, A_dyn


def make_definite(A, sign="pos", floor=1e-6):
    """Nearest definite matrix to symmetric A: same eigenvectors, clipped eigenvalues."""
    w, U = np.linalg.eigh(0.5 * (A + A.T))       # real eigenvalues (ascending), orthonormal U
    thr = floor * np.abs(w).max()                # relative floor, not absolute
    if sign == "pos":
        w = np.clip(w, thr, None)                # all >= +thr  -> SPD
    else:                                        # sign == "neg"
        w = np.clip(w, None, -thr)               # all <= -thr  -> negative definite
    return U @ np.diag(w) @ U.T


def save_training_state(point, cost_history, grad_history, iter_history):
    os.makedirs(output_dir, exist_ok=True)
    Phi, A2 = point
    np.save(os.path.join(output_dir, "Phi_nit.npy"),     Phi)
    np.save(os.path.join(output_dir, "A2_nit.npy"),      A2)
    np.save(os.path.join(output_dir, "costvec_nit.npy"), np.asarray(cost_history))
    np.save(os.path.join(output_dir, "gradvec_nit.npy"), np.asarray(grad_history))
    np.save(os.path.join(output_dir, "itervec_nit.npy"), np.asarray(iter_history))


# ============================================================================
# Configuration
# ============================================================================
TESTCASE = "bracket"          # which testcase to run: "bracket" | "cantilever_plate"

# Initialization (basis and tensor seeds are chosen independently). The pair
# labels the run and selects its solution container output/<basis>_<tensor>
# (e.g. cl_opinf), so the four cases never overwrite each other.
BASIS   = "cl"                # reduced basis: "cl" (cotangent-lift) | "pod" | "random"
TENSOR  = "opinf"             # A2 seed:       "pg" | "opinf" | "random"
RESTART = True               # True: resume Phi_nit/A2_nit of THIS (BASIS, TENSOR) container

case       = get_case(TESTCASE)
data_dir   = case.data_dir
output_dir = nitrom_output_dir(case, BASIS, TENSOR)   # this run's container
ensure_nitrom_dirs(case)      # create any missing labeled containers up front

# ROM / cost
poly_comp    = [1]
leggauss_deg = 350
nsave_rom    = 5
which_fix    = "fix_none"      # "fix_bases" | "fix_tensors" | "fix_none"
# Internal symplectic step (None = one step per snapshot). Commensurate with the
# snapshot spacing (1e-4 = 4000 * dt_rom), so the cost/gradient sample the ROM at
# data times exactly on integration knots -- genuine symplectic states, no
# spline interpolation error (the spline between knots is not symplectic).
dt_rom       = 2.5e-8

# Optimizer
inner_iterations = 1000
trajectory_step  = 2          # snapshot increment between progressive phases
first_window     = 201         # initial training window (snapshots)

# ============================================================================
# MPI setup
# ============================================================================
# The training data is a single trajectory stored as one (states x times)
# matrix x.npy (velocities in xdot.npy). To parallelize, split it into one
# contiguous time segment per MPI rank; each segment is a short trajectory
# starting from its true FOM state (multiple-shooting). Run serially (or with
# `mpiexec -n 1`) and the whole trajectory is a single segment.
fname_x      = os.path.join(data_dir, "x_train.npy")    # TRAIN snapshots only (first 60%)
fname_xdot   = os.path.join(data_dir, "xdot_train.npy")
fname_time   = os.path.join(data_dir, "time_train.npy")
fname_weight = os.path.join(data_dir, "weight.npy")     # optional scalar weight

pool = classes.mpi_pool.from_single_trajectory(
    MPI.COMM_WORLD,
    fname_x,
    fname_time,
    n_segments=MPI.COMM_WORLD.Get_size(),
    fname_derivs=fname_xdot,
    fname_weights=fname_weight if os.path.exists(fname_weight) else None,
)
n_traj    = pool.n_traj
time      = pool.time
verbosity = 2 if pool.rank == 0 else 0

# ============================================================================
# Problem setup
# ============================================================================
fom = case.fom                                           # None: Hamiltonian cost needs no FOM

# Fresh start: seed (Phi, A2) from the chosen basis/tensor pair.
# Restart:     resume the previous NiTROM solution of this same container.
if pool.rank == 0:
    mode = "RESTART from previous run in" if RESTART else "fresh start ->"
    print(f"Initialization: BASIS={BASIS!r}  TENSOR={TENSOR!r}  ({mode} {output_dir})")

if RESTART:
    restart_point = os.path.join(output_dir, "Phi_nit.npy")
    if not os.path.exists(restart_point):
        raise FileNotFoundError(
            f"RESTART=True but no previous solution in {output_dir}; train the "
            f"(BASIS={BASIS!r}, TENSOR={TENSOR!r}) case once with RESTART=False first.")
    Phi_init = np.load(restart_point)
    A2_init  = np.load(os.path.join(output_dir, "A2_nit.npy"))
    if pool.rank == 0:
        print(f"Resumed: Phi {Phi_init.shape}, A2 {A2_init.shape}")
else:
    Phi_init   = init_basis(BASIS)
    A2_init, _ = init_tensor(TENSOR, Phi_init)

# Refuse a basis that cannot define a symplectic ROM: r must be EVEN (an odd
# skew Jhat is exactly singular -- eigenvalues come in +/-i*lambda pairs) and
# Jhat must be invertible. Checked on ALL ranks so an MPI run aborts
# consistently instead of deadlocking on a rank-0-only raise.
_Jhat0 = Phi_init.T @ np.asarray(canonical_J(Phi_init.shape[0]) @ Phi_init)
_cond0 = np.linalg.cond(_Jhat0)
if Phi_init.shape[1] % 2 or _cond0 > 1e12:
    raise ValueError(
        f"Jhat = Phi^T J Phi is (numerically) singular: r = {Phi_init.shape[1]}"
        f"{' (ODD -> singular by skew-symmetry)' if Phi_init.shape[1] % 2 else ''}, "
        f"cond(Jhat) = {_cond0:.3e}. Regenerate the basis (prepare_data.py caps "
        f"and evens the POD rank) before training.")

# Initial-point diagnostics (rank 0). cond(Jhat): canonical (cond = 1) for the
# cotangent lift; a large cond flags an ill-conditioned Jhat^-1 in the
# encoder/decoder and dynamics (typical for a plain POD basis). The spectrum of
# the ACTUAL initial NiTROM dynamics A = Jhat (A2 + A2^T) -- valid for every
# seed, incl. restart/random -- should sit on the imaginary axis (max Re ~ 0);
# max |Im| sets the fastest mode the gradient quadrature must resolve
# (leggauss_deg ~> |Im| * snapshot spacing / 2).
if pool.rank == 0:
    print(f"Init point: BASIS={BASIS!r} -> Phi {Phi_init.shape},  "
          f"TENSOR={TENSOR!r} -> A2 {A2_init.shape}, ||A2|| = {np.linalg.norm(A2_init):.3e}")
    Jhat_init = Phi_init.T @ np.asarray(canonical_J(Phi_init.shape[0]) @ Phi_init)
    print(f"cond(Jhat) = {np.linalg.cond(Jhat_init):.3e}   (canonical CL basis = 1)")
    ev = np.linalg.eigvals(Jhat_init @ (A2_init + A2_init.T))
    print(f"init dynamics A = Jhat (A2+A2^T): max Re = {ev.real.max():+.3e}, "
          f"max |Im| = {np.abs(ev.imag).max():.3e}")

n, r   = Phi_init.shape
St     = manifolds.Stiefel(n, r)
Euc_rr = manifolds.Euclidean(r, r)
M      = manifolds.Product([St, Euc_rr])
point  = (Phi_init, A2_init)

# ============================================================================
# Optimization setup
# ============================================================================
line_searcher = myAdaptiveLineSearcher(
    contraction_factor=0.5,
    sufficient_decrease=1e-4,
    max_iterations=25,
    initial_step_size=1.0e-2,
)

# Per-segment length can be small when many ranks split a short trajectory;
# clamp the initial window so the progressive schedule is never empty.
window0            = min(first_window, len(time))
trajectory_lengths = list(range(window0, len(time) + 1, trajectory_step))
if trajectory_lengths[-1] != len(time):
    trajectory_lengths.append(len(time))
outer_iterations   = len(trajectory_lengths)

cost_history  = []
grad_history  = []
iter_history  = []
iter_offset   = 0
error_metrics = []

# ============================================================================
# Training
# ============================================================================
if pool.rank == 0:
    fom_freq_path = os.path.join(data_dir, "fom_freqs.npy")
    if os.path.exists(fom_freq_path):
        w_min, w_max = np.load(fom_freq_path)
        nyq = np.pi / (time[1] - time[0])
        print(f"FOM dynamics S=J·A: |Im| in [{w_min:.3e}, {w_max:.3e}] rad/s  "
              f"(snapshot Nyquist ω={nyq:.3e}; fastest mode {w_max / nyq:.0f}× over-Nyquist)")
    print("=" * 70)
    print(f"{f'Bracket  ·  Hamiltonian NiTROM Training  ·  case {BASIS}_{TENSOR}':^70}")
    print(f"  FOM state dim   2N = {n}")
    print(f"  reduced dim      r = {r}")
    print(f"  segments (= ranks)   {n_traj}   (snapshots/segment = {len(time)})")
    print(f"  progressive window   {trajectory_lengths[0]} -> {trajectory_lengths[-1]} "
          f"(+{trajectory_step}), {outer_iterations} phases")
    print(f"  optimizer            Conjugate Gradient, {inner_iterations} inner its, {which_fix}")
    print("=" * 70, flush=True)

t_start = tlib.perf_counter()

for phase, max_snapshot_idx in enumerate(trajectory_lengths):
    which_times = np.arange(min(max_snapshot_idx, len(time)))
    opt_obj     = make_opt_obj(which_fix, which_times)
    cost, grad, hess = nitrom_functions.create_objective_and_gradient(M, opt_obj, pool, fom)

    if phase == 0:
        initial_cost = cost(*point)
        if pool.rank == 0:
            print(f"Initial cost: {initial_cost:.6e}", flush=True)
    if pool.rank == 0:
        print(f"Phase {phase + 1}/{outer_iterations}: {len(which_times)} snapshots", flush=True)

    problem   = pymanopt.Problem(M, cost, euclidean_gradient=grad)
    optimizer = optimizers.ConjugateGradient(
        max_iterations=inner_iterations,
        min_step_size=1e-20,
        max_time=3600,
        line_searcher=line_searcher,
        log_verbosity=1,
        verbosity=verbosity,
    )

    result = optimizer.run(problem, initial_point=point)
    point  = result.point

    iterations  = result.log["iterations"]
    phase_costs = np.asarray(iterations["cost"])
    phase_grads = np.asarray(iterations["gradient_norm"])

    error_metrics.append({
        "phase":           phase,
        "length":          len(which_times),
        "initial_cost":    phase_costs[0],
        "final_cost":      phase_costs[-1],
        "final_grad_norm": phase_grads[-1],
    })

    iter_history.extend(np.arange(len(phase_costs)) + iter_offset)
    cost_history.extend(phase_costs)
    grad_history.extend(phase_grads)
    iter_offset = iter_history[-1]

    if pool.rank == 0:
        save_training_state(point, cost_history, grad_history, iter_history)

# ============================================================================
# Final reporting + plot (rank 0)
# ============================================================================
if pool.rank == 0:
    elapsed = tlib.perf_counter() - t_start
    print(f"Done in {elapsed:.1f} s", flush=True)
    np.save(os.path.join(output_dir, "error_metrics.npy"), error_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(iter_history, cost_history, "k-", lw=2)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Cost"); axes[0].set_title("Cost")
    axes[1].semilogy(iter_history, grad_history, "k-", lw=2)
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel("Gradient norm"); axes[1].set_title("Gradient")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progression.png"), dpi=150)
    plt.close()
