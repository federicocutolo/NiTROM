"""
POD rank-truncation sweep for the wave-equation ROMs.

For each spatial POD rank in RANKS, rebuild the cotangent-lift basis, recompute
the VC-H-OpInf and Petrov-Galerkin baselines, retrain NiTROM (warm-started from
OpInf at that rank), and score ALL THREE ROMs with the SAME global (Frobenius)
relative error on the HELD-OUT test set over the training horizon [0, T]:

    e = sqrt( sum_j ||X_j - Xhat_j||_F^2 / sum_j ||X_j||_F^2 )

NiTROM is scored by this same relative error as the baselines -- NOT its training
cost, which is a different (weighted, training-set) objective and not comparable.
Then plots e vs. the reduced dimension r = 2 x rank.

This script is self-contained and INDEPENDENT of train_nitrom.py: it drives the
reusable core in nitrom_pipeline.py directly.

Restartable: per-rank (Phi_nit, A2_nit) are cached under output_dir/rank_sweep/;
a rank with a cache is reused instead of retrained (set FORCE_RETRAIN to redo).

Run:
    python rank_sweep.py                         # serial
    mpiexec -n N python -u rank_sweep.py         # MPI (N <= n_traj)
Quick smoke pass:
    SWEEP_RANKS=4,6 SWEEP_INNER=10 python rank_sweep.py
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymanopt.manifolds as manifolds
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from NiTROM.Optimization_Functions import classes
from NiTROM.PyManopt_Functions.my_pymanopt_classes import myAdaptiveLineSearcher
from wave_cases import get_case
import wave_roms
import wave_sampling
from nitrom_pipeline import (run_training, global_relative_error,
                             galerkin_maps, symplectic_maps)
import seaborn as sns
sns.set(style="ticks", font_scale=1.2, palette="tab10")


# ============================================================================
# Configuration
# ============================================================================
TESTCASE      = "dirichlet"
# Spatial POD ranks to sweep (reduced dim = 2 x rank). Override for a quick pass.
RANKS         = [int(x) for x in os.environ.get("SWEEP_RANKS", "4, 6, 8, 10, 12, 14").split(",")]
FORCE_RETRAIN = True                      # ignore cached per-rank solutions

# Held-out test set (same sampling space as training, different seed).
TEST_SEED = 7
N_TEST    = 10

# NiTROM training hyperparameters (independent copy; mirrors train_nitrom.py).
poly_comp        = [1]
leggauss_deg     = 5
nsave_rom        = 5
which_fix        = "fix_none"
dt_rom           = 1e-2
inner_iterations = int(os.environ.get("SWEEP_INNER", "500"))
first_window     = 100
trajectory_step  = 50


# ============================================================================
# Setup (paths, MPI pool, data, test set) -- all rank-independent
# ============================================================================
case      = get_case(TESTCASE)
fom       = case.fom
data_dir  = case.data_dir
out_dir   = case.output_dir
test_out  = case.test_results_dir
sweep_dir = os.path.join(out_dir, "rank_sweep")
os.makedirs(sweep_dir, exist_ok=True)
os.makedirs(test_out,  exist_ok=True)

# MPI pool over the (fixed) training trajectories -- built once, reused per rank.
fname_traj   = os.path.join(data_dir, "traj_%03d.npy")
fname_time   = os.path.join(data_dir, "time.npy")
fname_weight = os.path.join(data_dir, "weight_%03d.npy")
n_traj = len([f for f in os.listdir(data_dir)
              if f.startswith("traj_") and f.endswith(".npy")])
if n_traj == 0:
    raise ValueError(f"No trajectory files in {data_dir}. Run main.py first.")
pool      = classes.mpi_pool(MPI.COMM_WORLD, n_traj, fname_traj, fname_time,
                             fname_weights=fname_weight)
time      = pool.time
rank0     = pool.rank == 0
verbosity = 2 if rank0 else 0

# Training snapshots (full state) for the basis + the OpInf operator.
traj_files  = sorted(f for f in os.listdir(data_dir) if f.startswith("traj_")  and f.endswith(".npy"))
deriv_files = sorted(f for f in os.listdir(data_dir) if f.startswith("deriv_") and f.endswith(".npy"))
if len(deriv_files) != len(traj_files):
    raise FileNotFoundError(f"{len(traj_files)} traj_*.npy but {len(deriv_files)} deriv_*.npy in "
                            f"{data_dir}; rerun main.py to regenerate velocity snapshots.")
train_trajs  = [np.load(os.path.join(data_dir, f)) for f in traj_files]
train_derivs = [np.load(os.path.join(data_dir, f)) for f in deriv_files]
X_mat   = np.stack(train_trajs, 0).transpose(1, 0, 2).reshape(2 * fom.N, -1)
W_fom   = np.load(os.path.join(data_dir, "W_fom.npy"))
J       = wave_roms.canonical_J(2 * fom.N)

# Held-out test trajectories on the training-horizon grid `time` (the FOM
# reference the relative error is measured against).
test_trajs, _, _ = wave_sampling.generate_trajectories(
    fom, time, N_TEST, seed=TEST_SEED, ranges=case.sample_ranges)
test_trajs = list(test_trajs)

line_searcher = myAdaptiveLineSearcher(
    contraction_factor=0.5,
    sufficient_decrease=1e-4,
    max_iterations=25,
    initial_step_size=1.0e-2,
)

if rank0:
    print("=" * 70)
    print(f"{'POD rank-truncation sweep  ·  ' + case.label:^70}")
    print("=" * 70)
    print(f"  ranks            : {RANKS}   (reduced dim = 2 x rank)")
    print(f"  training trajs   : {n_traj} x {len(time)} snapshots")
    print(f"  test trajs       : {N_TEST} (seed {TEST_SEED}), horizon [0, {time[-1]:.2f}]")
    print(f"  inner iterations : {inner_iterations}")
    print("=" * 70, flush=True)


# ============================================================================
# Sweep
# ============================================================================
results = []
for rank in RANKS:
    Phi, _, _ = wave_roms.cotangent_lift_basis(X_mat, fom.N, rank=rank)
    n, r = Phi.shape                                    # r = 2 * rank

    # Baselines (pure functions of Phi) ------------------------------------------
    A_bar       = wave_roms.vc_h_opinf_operator(Phi, J, train_trajs, train_derivs)
    A_dyn_opinf = wave_roms.opinf_dynamics(Phi, J, A_bar)
    A_dyn_PG    = wave_roms.pg_dynamics(Phi, J, W_fom)

    # NiTROM: reuse cached solution if present, else train ------------------------
    rdir  = os.path.join(sweep_dir, f"rank_{rank:03d}")
    phi_f = os.path.join(rdir, "Phi_nit.npy")
    a2_f  = os.path.join(rdir, "A2_nit.npy")
    if (not FORCE_RETRAIN) and os.path.exists(phi_f) and os.path.exists(a2_f):
        Phi_nit, A2_nit = np.load(phi_f), np.load(a2_f)
        if rank0:
            print(f"\n[rank {rank} | rdim {r}] using cached NiTROM solution")
    else:
        A2_init = wave_roms.opinf_init_A2(Phi, J, A_bar)
        M = manifolds.Product([manifolds.Stiefel(n, r), manifolds.Euclidean(r, r)])
        if rank0:
            print(f"\n=== Training NiTROM at rank {rank} (reduced dim {r}) ===", flush=True)
        res = run_training(
            pool, fom, M, (Phi.copy(), A2_init),
            inner_iterations=inner_iterations, first_window=first_window,
            trajectory_step=trajectory_step, which_fix=which_fix,
            leggauss_deg=leggauss_deg, nsave_rom=nsave_rom, poly_comp=poly_comp,
            dt_rom=dt_rom, line_searcher=line_searcher, verbosity=verbosity,
        )
        Phi_nit, A2_nit = res["point"]
        if rank0:
            os.makedirs(rdir, exist_ok=True)
            np.save(phi_f, Phi_nit)
            np.save(a2_f, A2_nit)
            np.save(os.path.join(rdir, "costvec_nit.npy"), np.asarray(res["cost_history"]))
            print(f"  trained in {res['elapsed']:.1f} s, final cost "
                  f"{res['error_metrics'][-1]['final_cost']:.3e}", flush=True)
    A_dyn_nit = (Phi_nit.T @ J @ Phi_nit) @ (A2_nit + A2_nit.T)      # Jhat_nit @ A2_nit

    # Global relative error on the test set, training horizon -- same metric/model
    e_opinf = global_relative_error(test_trajs, *galerkin_maps(Phi),       A_dyn_opinf, time)
    e_pg    = global_relative_error(test_trajs, *symplectic_maps(Phi, J),  A_dyn_PG,    time)
    e_nit   = global_relative_error(test_trajs, *symplectic_maps(Phi_nit, J), A_dyn_nit, time)
    results.append({"rank": rank, "rdim": r,
                    "e_opinf": e_opinf, "e_pg": e_pg, "e_nit": e_nit})
    if rank0:
        print(f"[rank {rank} | rdim {r}]  Global relative error - OpInf={e_opinf:.3e}  "
              f"PG={e_pg:.3e}  NiTROM={e_nit:.3e}", flush=True)


# ============================================================================
# Save + plot (rank 0)
# ============================================================================
if rank0:
    np.save(os.path.join(sweep_dir, "rank_sweep_results.npy"), results)

    rdims = [d["rdim"] for d in results]
    tab10 = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.semilogy(rdims, [d["e_pg"]    for d in results], "s-", color=tab10[1], label="Petrov-Galerkin")
    ax.semilogy(rdims, [d["e_opinf"] for d in results], "o-", color=tab10[2], label="VC-H-OpInf")
    ax.semilogy(rdims, [d["e_nit"]   for d in results], "^-", color=tab10[3], label="NiTROM")
    ax.set_xlabel(r"Basis dimension $r = 2 \times$ POD rank")
    ax.set_ylabel("relative error")
    ax.set_title("ROM relative error vs basis truncation rank")
    # ax.grid(True, which="both", alpha=0.3)
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    png = os.path.join(test_out, "rank_sweep_error.png")
    fig.savefig(png, dpi=150)
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {png}")
    print(f"Saved {os.path.join(sweep_dir, 'rank_sweep_results.npy')}")
