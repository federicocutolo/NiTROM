"""
Check that the cost and gradient are invariant to the number of MPI ranks.

Trajectories are split across ranks and the results allgathered+summed, so the
TOTAL cost/gradient must not depend on the rank count (only the floating-point
summation order changes -> agreement to ~1e-12, not bit-exact).

Workflow (n_traj must be >= number of ranks):
    python                test_parallel_consistency.py     # serial -> writes reference
    mpiexec -n 2 python   test_parallel_consistency.py
    mpiexec -n 5 python   test_parallel_consistency.py
    mpiexec -n 10 python  test_parallel_consistency.py

Each run evaluates (cost, grad) at the SAME deterministic point and compares
against the serial reference, printing relative deviations and PASS/FAIL.
"""
import os
import sys
import numpy as np
from mpi4py import MPI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pymanopt.manifolds as manifolds
from NiTROM.Optimization_Functions import classes, nitrom_functions
from main import WaveEquationFOM

# ---- fixed problem setup (identical on every rank and every run) ----
N, L, c       = 128, 2.0, 1.0
leggauss_deg  = 5
nsave_rom     = 5
poly_comp     = [1]
dt_rom        = 1e-4
window        = 20                 # fixed snapshot window
TOL           = 1e-9              # rel. tolerance (fp summation reorder ~1e-12)

data_dir = os.path.join(HERE, "trajectories")
res_dir  = os.path.join(HERE, "test_results")
os.makedirs(res_dir, exist_ok=True)

fom = WaveEquationFOM(N=N, L=L, c=c)

fname_traj   = os.path.join(data_dir, "traj_%03d.npy")
fname_time   = os.path.join(data_dir, "time.npy")
fname_weight = os.path.join(data_dir, "weight_%03d.npy")
n_traj = len([f for f in os.listdir(data_dir)
              if f.startswith("traj_") and f.endswith(".npy")])

pool = classes.mpi_pool(MPI.COMM_WORLD, n_traj, fname_traj, fname_time,
                        fname_weights=fname_weight)

# ---- deterministic evaluation point: POD basis + Petrov-Galerkin A2 ----
Phi  = np.load(os.path.join(data_dir, "Phi_pod.npy"))
n, r = Phi.shape
M    = manifolds.Product([manifolds.Stiefel(n, r), manifolds.Euclidean(r, r)])

half     = n // 2
J        = np.block([[np.zeros((half, half)), np.eye(half)],
                     [-np.eye(half),          np.zeros((half, half))]])
Jhat_inv = np.linalg.inv(Phi.T @ J @ Phi)
A2       = -Jhat_inv @ (Phi.T @ fom.W @ Phi) @ Jhat_inv   # deterministic, rank-independent

# ---- evaluate cost & gradient (both allgather internally) ----
opt_obj = classes.optimization_objects(
    pool, which_trajs=np.arange(pool.my_n_traj), which_times=np.arange(window),
    leggauss_deg=leggauss_deg, nsave_rom=nsave_rom, poly_comp=poly_comp,
    hamiltonian=True, which_fix="fix_none", dt_rom=dt_rom,
)
cost, grad, _ = nitrom_functions.create_objective_and_gradient(M, opt_obj, pool, fom)

J_cost    = cost(Phi, A2)
gPhi, gA2 = grad(Phi, A2)

# ---- record (rank 0) and compare to the serial reference ----
if pool.rank == 0:
    out = os.path.join(res_dir, f"parcheck_size{pool.size:02d}.npz")
    np.savez(out, size=pool.size, cost=J_cost, gPhi=gPhi, gA2=gA2)
    print(f"[size {pool.size}] cost={J_cost:.12e}  "
          f"||gPhi||={np.linalg.norm(gPhi):.12e}  ||gA2||={np.linalg.norm(gA2):.12e}")

    ref_path = os.path.join(res_dir, "parcheck_size01.npz")
    if pool.size == 1:
        print("Serial reference written. Now rerun with mpiexec -n K.")
    elif os.path.exists(ref_path):
        ref = np.load(ref_path)
        def reldiff(a, b):
            d = np.linalg.norm(np.asarray(a) - np.asarray(b))
            return d / max(np.linalg.norm(np.asarray(b)), 1e-300)
        dc = abs(J_cost - float(ref["cost"])) / max(abs(float(ref["cost"])), 1e-300)
        dP = reldiff(gPhi, ref["gPhi"])
        dA = reldiff(gA2,  ref["gA2"])
        worst = max(dc, dP, dA)
        print(f"[size {pool.size} vs serial]  rel cost {dc:.2e} | "
              f"rel gPhi {dP:.2e} | rel gA2 {dA:.2e}")
        print("PASS" if worst < TOL else f"FAIL (worst {worst:.2e} > {TOL:.0e})")
    else:
        print("No serial reference found -- run `python test_parallel_consistency.py` first.")
