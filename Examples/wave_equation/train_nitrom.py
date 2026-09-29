"""
Train a Hamiltonian NiTROM on the 1D linear wave equation.

Run:
    python  train_nitrom.py                          # serial
    mpiexec -n N python -u train_nitrom.py           # MPI (N <= n_traj)
"""

import os

# Must run before numpy import: pin BLAS/OpenMP threads per process.
def set_cpu_threads(n):
    n = str(int(n))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = n

set_cpu_threads(1)

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import sys
import time as tlib
import datetime
import numpy as np
import matplotlib.pyplot as plt

import pymanopt
import pymanopt.manifolds  as manifolds
import pymanopt.optimizers as optimizers
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from NiTROM.Optimization_Functions       import classes, nitrom_functions
from NiTROM.PyManopt_Functions.my_pymanopt_classes import myAdaptiveLineSearcher
from wave_cases import get_case, nitrom_output_dir, ensure_nitrom_dirs
import wave_roms


# ============================================================================
# Testcase selection
# ============================================================================
TESTCASE = "dirichlet"  # 'dirichlet' or 'periodic'  (see wave_cases.py)


# ============================================================================
# Helpers
# ============================================================================
def find_traj(data_dir):
    fname_traj   = os.path.join(data_dir, "traj_%03d.npy")
    fname_time   = os.path.join(data_dir, "time.npy")
    fname_weight = os.path.join(data_dir, "weight_%03d.npy")
    n_traj = len([f for f in os.listdir(data_dir)
                  if f.startswith("traj_") and f.endswith(".npy")])
    if n_traj == 0:
        raise ValueError(f"No trajectory files in {data_dir}. Run main.py first.")
    return fname_time, fname_traj, fname_weight, n_traj


def make_opt_obj(which_fix, which_times):
    # Pass which_times into the constructor so the cost weights are scaled by the
    # ACTUAL windowed snapshot count (n_snapshots): the constructor does
    # weights *= sum(counts)*n_snapshots, making the cost a mean error per snapshot.
    # Slicing X/time after construction (the old approach) left the weights scaled
    # by the full snapshot count, so the cost grew with the training window.
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

def load_train_snapshots(data_dir):
    """Trajectories and exact velocity snapshots (Xdot = J W X) for OpInf."""
    traj_files  = sorted(f for f in os.listdir(data_dir)
                         if f.startswith("traj_")  and f.endswith(".npy"))
    deriv_files = sorted(f for f in os.listdir(data_dir)
                         if f.startswith("deriv_") and f.endswith(".npy"))
    if len(deriv_files) != len(traj_files):
        raise FileNotFoundError(
            f"{len(traj_files)} traj_*.npy but {len(deriv_files)} deriv_*.npy in "
            f"{data_dir}; rerun main.py to regenerate velocity snapshots.")
    trajectories = [np.load(os.path.join(data_dir, f)) for f in traj_files]
    derivatives  = [np.load(os.path.join(data_dir, f)) for f in deriv_files]
    return trajectories, derivatives


def init_basis(basis):
    """Load the reduced basis Phi (both are written by main.py):
      "cl"  -> cotangent-lift symplectic basis (Jhat canonical, cond = 1)
      "pod" -> plain POD basis                 (Jhat generic, invertible)"""
    if basis not in ("cl", "pod"):
        raise ValueError(f"unknown BASIS={basis!r}; use 'cl' or 'pod'")
    Phi = np.load(os.path.join(data_dir, f"Phi_{basis}.npy"))
    if pool.rank == 0:
        print(f"Basis : {basis:6s} -> Phi {Phi.shape}")
    return Phi


def init_tensor(tensor, Phi):
    """Seed the reduced operator A2 for a GIVEN basis Phi:
      "pg"     -> Petrov-Galerkin projection of W_fom
      "opinf"  -> VC-H-OpInf (Lyapunov solve on the reduced snapshots)
      "random" -> small random perturbation (cold start)

    The PG/OpInf operators (and A2 conversions) come from ``wave_roms`` --
    basis-general formulas, so any BASIS x TENSOR pair is valid."""
    J     = wave_roms.canonical_J(Phi.shape[0])
    A_dyn = None
    if tensor == "pg":
        W_fom   = np.load(os.path.join(data_dir, "W_fom.npy"))   # H = 0.5 z^T W z
        A2_init = wave_roms.pg_init_A2(Phi, J, W_fom)
        A_dyn   = wave_roms.pg_dynamics(Phi, J, W_fom)
    elif tensor == "opinf":
        trajs, derivs = load_train_snapshots(data_dir)
        A_bar   = wave_roms.vc_h_opinf_operator(Phi, J, trajs, derivs)
        A2_init = wave_roms.opinf_init_A2(Phi, J, A_bar)
        A_dyn   = wave_roms.opinf_dynamics(Phi, J, A_bar)
    elif tensor == "random":
        r       = Phi.shape[1]
        A2_init = np.empty((r, r))
        if pool.rank == 0:
            A2_init[:] = np.random.randn(r, r) * 1e-4
        pool.comm.Bcast(A2_init, root=0)                         # same seed on all ranks
    else:
        raise ValueError(f"unknown TENSOR={tensor!r}; use 'pg', 'opinf', or 'random'")

    if pool.rank == 0:
        print(f"Tensor: {tensor:6s} -> A2 {A2_init.shape}, asymmetry "
              f"{np.linalg.norm(A2_init - A2_init.T):.3e}")
        if A_dyn is not None:
            eig = np.linalg.eigvals(A_dyn)
            print(f"        max Re(A_dyn) = {eig.real.max():+.3e}   "
                  f"max |Im(A_dyn)| = {np.abs(eig.imag).max():.3e}")
    return A2_init


def save_training_state(point, cost_history, grad_history, iter_history):
    os.makedirs(output_dir, exist_ok=True)
    Phi, A2 = point
    np.save(os.path.join(output_dir, "Phi_nit.npy"),     Phi)
    np.save(os.path.join(output_dir, "A2_nit.npy"),      A2)
    np.save(os.path.join(output_dir, "costvec_nit.npy"), np.asarray(cost_history))
    np.save(os.path.join(output_dir, "gradvec_nit.npy"), np.asarray(grad_history))
    np.save(os.path.join(output_dir, "itervec_nit.npy"), np.asarray(iter_history))
    np.savetxt(os.path.join(output_dir, "costvec_nit.txt"), np.asarray(cost_history), fmt="%.6e")
    np.savetxt(os.path.join(output_dir, "gradvec_nit.txt"), np.asarray(grad_history), fmt="%.6e")
    np.savetxt(os.path.join(output_dir, "itervec_nit.txt"), np.asarray(iter_history), fmt="%d")


# ============================================================================
# Configuration
# ============================================================================
case = get_case(TESTCASE)

# Paths
data_dir   = case.data_dir
# output_dir is set below, after BASIS/TENSOR are chosen (labeled container)

# FOM physics
N, L, c = case.N, case.L, case.c

# ROM / cost
poly_comp    = [1]
leggauss_deg = 350
nsave_rom    = 5
which_fix    = "fix_none"     # "fix_bases" | "fix_tensors" | "fix_none"
dt_rom       = 1e-4          # internal symplectic step (None = one step per snapshot)

# Initialization: BASIS and TENSOR are chosen INDEPENDENTLY (mirrors the
# bracket example); the solution is written to the labeled container
# output/<BASIS>_<TENSOR>/ so the four cases never overwrite each other.
BASIS   = "cl"        # reduced basis: "cl" (cotangent lift) | "pod"
TENSOR  = "opinf"     # A2 seed:       "pg" | "opinf" | "random"
RESTART = True       # resume THIS (BASIS, TENSOR) container's Phi_nit/A2_nit

# Labeled output container for this initialization case (overrides the flat
# case.output_dir): output/<BASIS>_<TENSOR>/.
output_dir = nitrom_output_dir(case, BASIS, TENSOR)
ensure_nitrom_dirs(case)

# Optimizer
inner_iterations = 1000
trajectory_step  = 5         # snapshot increment between progressive phases
first_window     = 100         # initial training window (snapshots)

# ============================================================================
# MPI setup
# ============================================================================
fname_time, fname_traj, fname_weight, n_traj = find_traj(data_dir)

pool = classes.mpi_pool(
    MPI.COMM_WORLD,
    n_traj,
    fname_traj,
    fname_time,
    fname_weights=fname_weight,
)
time      = pool.time
verbosity = 2 if pool.rank == 0 else 0


# ============================================================================
# Problem setup
# ============================================================================
fom = case.fom

if RESTART:
    Phi_init = np.load(os.path.join(output_dir, "Phi_nit.npy"))
    A2_init  = np.load(os.path.join(output_dir, "A2_nit.npy"))
    if pool.rank == 0:
        print(f"Init: RESTART from {output_dir} -> Phi {Phi_init.shape}, "
              f"A2 {A2_init.shape}")
else:
    if pool.rank == 0:
        print(f"Initialization: BASIS={BASIS!r}  TENSOR={TENSOR!r} "
              f"-> container {output_dir}")
    Phi_init = init_basis(BASIS)
    A2_init  = init_tensor(TENSOR, Phi_init)

# Refuse a basis that cannot define a symplectic ROM: r must be EVEN (an odd
# skew Jhat is exactly singular) and Jhat invertible. Checked on ALL ranks.
_Jhat0 = Phi_init.T @ wave_roms.canonical_J(Phi_init.shape[0]) @ Phi_init
_cond0 = np.linalg.cond(_Jhat0)
if Phi_init.shape[1] % 2 or _cond0 > 1e12:
    raise ValueError(
        f"Jhat = Phi^T J Phi is (numerically) singular: r = {Phi_init.shape[1]}"
        f"{' (ODD -> singular by skew-symmetry)' if Phi_init.shape[1] % 2 else ''}, "
        f"cond(Jhat) = {_cond0:.3e}. Regenerate the bases with main.py.")

n, r   = Phi_init.shape                     # manifold sized to the CHOSEN basis
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

# The progressive window runs from first_window up to the FULL trajectory length,
# which is the snapshot-column count of the loaded data (len(time) == traj.shape[1],
# NOT x.npy -- in the wave case x.npy is the spatial grid). Clamp first_window to
# that count so an oversized value (e.g. for a restart) collapses to a single
# full-window phase instead of producing an empty schedule (IndexError below).
n_snapshots = len(time)
if first_window > n_snapshots:
    if pool.rank == 0:
        print(f"[warn] first_window={first_window} > snapshots/traj={n_snapshots}; "
              f"clamping to {n_snapshots} (single full-window phase).")
    first_window = n_snapshots
trajectory_lengths = list(range(first_window, n_snapshots + 1, trajectory_step))
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
    sep, rule = "=" * 70, "-" * 70

    def _section(title):
        print(rule)
        print(f"  {title}")
        print(rule)

    def _row(label, value):
        print(f"    {label:<20}{value}")

    print(sep)
    print(f"{'Wave Equation  ·  Hamiltonian NiTROM Training':^70}")

    _section("Problem size")
    _row("FOM state dim",  f"n = {n}   (N = {N} nodes per field)")
    _row("reduced dim",    f"r = {r}")
    _row("Hamiltonian",    f"poly_comp = {poly_comp}")
    _row("search manifold", f"Stiefel({n}, {r})  x  Euclidean({r}, {r})")

    _section("Training data")
    _row("trajectories",     n_traj)
    _row("snapshots / traj", len(time))
    _row("source",           data_dir)

    _section("Initialization")
    _basis_desc  = {"cl": "cotangent lift", "pod": "plain POD"}.get(BASIS, BASIS)
    _tensor_desc = {"pg": "Petrov-Galerkin", "opinf": "VC-H-OpInf",
                    "random": "random cold start"}.get(TENSOR, TENSOR)
    if RESTART:
        _row("mode",   f"RESTART  (resume Phi_nit / A2_nit)")
    else:
        _row("mode",   "fresh start")
    _row("basis",      f"{BASIS:6s} ({_basis_desc})")
    _row("tensor A2",  f"{TENSOR:6s} ({_tensor_desc})"
                       + ("   [ignored on restart]" if RESTART else ""))
    _row("container",  output_dir)

    _section("Progressive window")
    _row("start length", f"{trajectory_lengths[0]:>4d} snapshots")
    _row("increment",    f"{trajectory_step:>4d} snapshots")
    _row("final length", f"{trajectory_lengths[-1]:>4d} snapshots")
    _row("phases",       outer_iterations)

    _section("Optimization")
    _row("optimizer",        "Conjugate Gradient")
    _row("inner iterations", inner_iterations)
    _fix_desc = {"fix_none":    "nothing (bases + tensors both free)",
                 "fix_bases":   "bases    (Phi frozen; optimize A2 only)",
                 "fix_tensors": "tensors  (A2 frozen; optimize Phi only)"}.get(which_fix, "?")
    _row("which_fix",        f"{which_fix:12s} -> fixing {_fix_desc}")
    _row("line search",      f"adaptive  (init step {line_searcher._initial_step_size:.1e}, "
                             f"contraction {line_searcher._contraction_factor:g})")
    print(sep, flush=True)

t_start = tlib.perf_counter()

for phase, max_snapshot_idx in enumerate(trajectory_lengths):
    which_times = np.arange(min(max_snapshot_idx, len(time)))
    opt_obj     = make_opt_obj(which_fix, which_times) 
    # Define opt_obj here to pass number of snapshots according to the current training window  !!!!
    # So that cost AND gradient are corectly scaled correctly. 
    cost, grad, hess = nitrom_functions.create_objective_and_gradient(M, opt_obj, pool, fom)

    if phase == 0:
        initial_cost = cost(*point)
        if pool.rank == 0:
            print(f"Initial cost: {initial_cost:.6e}", flush=True)
    if pool.rank == 0:
        print(f"Phase {phase + 1}/{outer_iterations}: "
              f"{len(which_times)} snapshots", flush=True)
        
######## Difference between euclidean and riemaniann gradient and cost???????????!!!!!!!!!!!!!!!!!!!!!!!!!!!!!???????
    problem   = pymanopt.Problem(M, 
                                 cost, 
                                 euclidean_gradient=grad,) 
                                #  euclidean_hessian=hess,)

    optimizer = optimizers.ConjugateGradient(
        max_iterations=inner_iterations,
        min_step_size=1e-20,
        max_time=3600,
        line_searcher=line_searcher,
        log_verbosity=1,
        verbosity=verbosity,
    )

    # optimizer = optimizers.TrustRegions(
    #     miniter=3, 
    #     kappa=0.1, 
    #     theta=1.0, 
    #     rho_prime=0.1, 
    #     use_rand=False, 
    #     rho_regularization=1000.0,
    # )

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
# Final reporting
# ============================================================================
if pool.rank == 0:
    elapsed = tlib.perf_counter() - t_start
    fpath   = os.path.join(output_dir, "total_time.npy")
    prev    = float(np.load(fpath)) if (RESTART and os.path.exists(fpath)) else 0.0
    total_time = prev + elapsed
    print(f"This run: {elapsed:.1f} s | cumulative: {total_time:.1f} s", flush=True)
    np.save(fpath, total_time)
    np.save(os.path.join(output_dir, "error_metrics.npy"), error_metrics)



# ============================================================================
# Plots (rank 0 only)
# ============================================================================
if pool.rank == 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].semilogy(iter_history, cost_history, "k-", linewidth=2)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Cost")
    axes[0, 0].set_title("Training Cost Convergence")
    # Print training time
    elapsed_str = str(datetime.timedelta(seconds=int(total_time)))
    axes[0, 0].text(
        0.02, 0.05,
        f"Time: {elapsed_str}",
        transform=axes[0, 0].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )


    
    axes[0, 1].semilogy(iter_history, grad_history, "k-", linewidth=2)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Gradient Norm")
    axes[0, 1].set_title("Gradient Norm Convergence")

    phases      = [m["phase"]      for m in error_metrics]
    final_costs = [m["final_cost"] for m in error_metrics]
    lengths     = [m["length"]     for m in error_metrics]

    axes[1, 0].plot(phases, final_costs, "o-", color="k", markersize=8, linewidth=2)
    axes[1, 0].set_xlabel("Training Phase")
    axes[1, 0].set_ylabel("Final Cost")
    axes[1, 0].set_title("Cost Reduction Across Phases")

    axes[1, 1].bar(phases, lengths, color="k", edgecolor="k")
    axes[1, 1].set_xlabel("Training Phase")
    axes[1, 1].set_ylabel("Trajectory Length (snapshots)")
    axes[1, 1].set_title("Progressive Trajectory Extension")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progression.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "training_progression.pdf"), bbox_inches="tight")
    plt.close()
