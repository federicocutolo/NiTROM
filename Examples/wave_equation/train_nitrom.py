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
from scipy.sparse import bmat, eye, csr_matrix

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
from wave_cases import get_case


# ============================================================================
# Testcase selection
# ============================================================================
TESTCASE = "periodic"  # 'dirichlet' or 'periodic'  (registry lives in wave_cases.py)


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

def initialize_training(init):
    """
    Seed (Phi, A2) before optimization. ``init`` picks both basis and tensors:
      "pg"      -> POD basis + Petrov-Galerkin  warm-start A2
      "opinf"   -> POD basis + VarCon-H-OpInf   warm-start A2
      "random"  -> POD basis + small random A2          (cold start)
      "restart" -> continue a previous NiTROM run (Phi_nit, A2_nit)
    """
    if init in ("pg", "opinf"):
        Phi_init = np.load(os.path.join(data_dir, "Phi_cl.npy"))

        # Reduced symplectic structure built from the basis.
        half_dim = Phi_init.shape[0] // 2
        J = np.block([[np.zeros((half_dim, half_dim)), np.eye(half_dim)],
                      [-np.eye(half_dim), np.zeros((half_dim, half_dim))]])
        print("J shape: ", J.shape)
        Jhat = Phi_init.T @ J @ Phi_init
        Jhat_inv = np.linalg.inv(Jhat)
        print("Jhat shape: ", Jhat.shape)
        sym = np.linalg.norm(Jhat + Jhat.T)
        print(f"Check Jhat skew-symmetry: {sym:.6e}  (~0)")

        # Petrov-Galerkin warm-start: A2_init = -Jhat^{-1} W_hat Jhat^{-1}.
        W_fom    = np.load(os.path.join(data_dir, "W_fom.npy"))    # H = 0.5 z^T W z
        W_hat    = Phi_init.T @ W_fom @ Phi_init # W fom either from npy or from fom class
        A2_PG      = Jhat_inv @ W_hat @ Jhat
        A_dyn_PG   = Jhat @ A2_PG
        eig_pg     = np.linalg.eigvals(A_dyn_PG)
        print("max real part (A_dyn_PG):", eig_pg.real.max())
        print("max |imag| (A_dyn_PG):   ", np.abs(eig_pg.imag).max())
        print(f"  asymmetry A2_init_PG = {np.linalg.norm(A2_PG - A2_PG.T):.6e}  (~0)")

        # OpInf warm-start: A_bar from test_nitrom's VC-H-OpInf solve. Convert to NiTROM's
        # A2 parameterization: OpInf says z_dot = Jhat^{-T} A_bar z; NiTROM says z_dot =
        # 2 Jhat A2 z. Equating gives  A2 = -1/2 * Jhat^{-2} * A_bar (= A_bar/2 if Jhat canonical).   
        A_bar       = np.load(os.path.join(output_dir, "A_bar_opinf.npy"))
        A_dyn_opinf = Jhat_inv.T @ A_bar
        A2_opinf    = 0.5 * A_bar # 0.5 * np.linalg.inv(Jhat.T @ Jhat) @ A_bar

        eig_op        = np.linalg.eigvals(A_dyn_opinf)
        print("Check norm 0.5A_bar - A2_init_opinf: ", np.linalg.norm(0.5 * A_bar - A2_opinf))

        print("max real part (A_dyn_opinf):", eig_op.real.max())
        print("max |imag| (A_dyn_opinf):   ", np.abs(eig_op.imag).max())
        print(f"  asymmetry A2_init_opinf = {np.linalg.norm(A2_opinf - A2_opinf.T):.6e}  (~0)")

        A2_init = A2_PG.copy() if init == "pg" else A2_opinf.copy()
        if pool.rank == 0:
            print(f"Init: POD basis + {init} warm-start A2")

    elif init == "random":
        # Cold start: POD basis + small random A2 (broadcast for rank consistency).
        Phi_init = np.load(os.path.join(data_dir, "Phi_cl.npy"))
        r        = Phi_init.shape[1]
        A2_init  = np.empty((r, r))
        if pool.rank == 0:
            A2_init[:] = np.random.randn(r, r) * 1e-4
        pool.comm.Bcast(A2_init, root=0)
        if pool.rank == 0:
            print("Init: POD basis + small random A2")

    elif init == "restart":
        # Continue a previous NiTROM solution (both basis and tensors).
        Phi_init = np.load(os.path.join(output_dir, "Phi_nit.npy"))
        A2_init  = np.load(os.path.join(output_dir, "A2_nit.npy"))
        if pool.rank == 0:
            print("Init: restart from previous NiTROM solution")

    else:
        raise ValueError(
            f"unknown INIT={init!r}; use 'pg', 'opinf', 'random', or 'restart'"
        )

    return Phi_init, A2_init


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
output_dir = case.output_dir

# FOM physics
N, L, c = case.N, case.L, case.c

# ROM / cost
poly_comp    = [1]
leggauss_deg = 5
nsave_rom    = 5
which_fix    = "fix_none"     # "fix_bases" | "fix_tensors" | "fix_none"
dt_rom       = 1e-4           # internal symplectic step (None = one step per snapshot)

# Initialization: how to seed (Phi, A2) before optimizing.
#   "pg"      -> POD basis + Petrov-Galerkin  warm-start A2
#   "opinf"   -> POD basis + VarCon-H-OpInf   warm-start A2
#   "random"  -> POD basis + small random A2          (cold start)
#   "restart" -> continue a previous NiTROM solution
INIT = "opinf"

# Optimizer
inner_iterations = 600
trajectory_step  = 2         # snapshot increment between progressive phases
first_window     = 2         # initial training window (snapshots)

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

Phi_pod = np.load(os.path.join(data_dir, "Phi_cl.npy"))
print(f"Loaded POD basis from {data_dir}, shape: {Phi_pod.shape}")
n, r    = Phi_pod.shape

St     = manifolds.Stiefel(n, r)
Euc_rr = manifolds.Euclidean(r, r)
M      = manifolds.Product([St, Euc_rr])


## Hard-code initialization of A2_init = \Phi.T W_fom \Phi 
# half_dim = n // 2
# J = bmat([[csr_matrix((half_dim, half_dim)), eye(half_dim)],
#           [-eye(half_dim), csr_matrix((half_dim, half_dim))]
#           ])

# W = np.load(os.path.join(data_dir, "W_fom.npy"))
# print("W FOM: ", W)
# A2_init_Galerkin = 0.5 * Phi_pod.T @ J @ W @ Phi_pod
# print("A2_init_Galerkin size: ", A2_init_Galerkin.shape)


Phi_init, A2_init = initialize_training(INIT)
point = (Phi_init, A2_init)


# ============================================================================
# Optimization setup
# ============================================================================
line_searcher = myAdaptiveLineSearcher(
    contraction_factor=0.5,
    sufficient_decrease=0.08,
    max_iterations=25,
    initial_step_size=1.0e-5,
)

trajectory_lengths = list(range(first_window, len(time) + 1, trajectory_step))
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

    _section("Progressive window")
    _row("start length", f"{trajectory_lengths[0]:>4d} snapshots")
    _row("increment",    f"{trajectory_step:>4d} snapshots")
    _row("final length", f"{trajectory_lengths[-1]:>4d} snapshots")
    _row("phases",       outer_iterations)

    _section("Optimization")
    _row("optimizer",        "Conjugate Gradient")
    _row("inner iterations", inner_iterations)
    _row("which_fix",        which_fix)
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
    prev    = float(np.load(fpath)) if (INIT == "restart" and os.path.exists(fpath)) else 0.0
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
