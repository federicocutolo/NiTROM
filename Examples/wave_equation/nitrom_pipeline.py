"""
Reusable NiTROM training + evaluation core for the wave-equation example.

These are the callable building blocks shared by train_nitrom.py (single run)
and rank_sweep.py (one run per POD truncation rank):

  * run_training        -- progressive-window Riemannian CG optimization of (Phi, A2)
  * global_relative_error -- aggregate (Frobenius) relative error of a ROM on a
                             set of FOM trajectories, the SAME metric for every model

Pure library: no module-level side effects, no fixed paths. The entry scripts
own configuration, I/O, BLAS-thread pinning, and plotting.
"""
import os
import sys
import time as tlib

import numpy as np
import pymanopt
import pymanopt.optimizers as optimizers
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from NiTROM.Optimization_Functions import classes, nitrom_functions


# ── Training ──────────────────────────────────────────────────────────────────
def run_training(pool, fom, M, point, *,
                 inner_iterations, first_window, trajectory_step,
                 which_fix, leggauss_deg, nsave_rom, poly_comp, dt_rom,
                 line_searcher, verbosity=0, save_cb=None):
    """Progressive-window Riemannian CG optimization of the NiTROM (Phi, A2).

    ``point`` is the (Phi_init, A2_init) warm start on the product manifold ``M``.
    The training window grows from ``first_window`` snapshots in ``trajectory_step``
    increments; each window rebuilds the cost so its weights are scaled by the
    actual windowed snapshot count (mean error per snapshot).

    Pure compute -- no file I/O or plotting. Pass ``save_cb(point, cost_history,
    grad_history, iter_history)`` to checkpoint after each phase (rank 0 only).
    Returns a dict: point, cost_history, grad_history, iter_history,
    error_metrics, elapsed.
    """
    time = pool.time

    def make_opt_obj(which_times):
        # which_times into the constructor so cost AND gradient are scaled by the
        # current window's snapshot count (not the full-trajectory count).
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

    trajectory_lengths = list(range(first_window, len(time) + 1, trajectory_step))
    cost_history, grad_history, iter_history, error_metrics = [], [], [], []
    iter_offset = 0
    t_start = tlib.perf_counter()

    for phase, max_snapshot_idx in enumerate(trajectory_lengths):
        which_times = np.arange(min(max_snapshot_idx, len(time)))
        opt_obj     = make_opt_obj(which_times)
        cost, grad, _ = nitrom_functions.create_objective_and_gradient(M, opt_obj, pool, fom)

        if phase == 0:
            # cost() runs a collective (allgather) internally, so EVERY rank must
            # call it -- evaluating on rank 0 only desynchronizes the MPI
            # collectives and corrupts every later cost/gradient reduction.
            init_cost = cost(*point)
            if pool.rank == 0:
                print(f"Initial cost: {init_cost:.6e}", flush=True)
        if pool.rank == 0:
            print(f"Phase {phase + 1}/{len(trajectory_lengths)}: "
                  f"{len(which_times)} snapshots", flush=True)

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

        if pool.rank == 0 and save_cb is not None:
            save_cb(point, cost_history, grad_history, iter_history)

    return {
        "point":         point,
        "cost_history":  cost_history,
        "grad_history":  grad_history,
        "iter_history":  iter_history,
        "error_metrics": error_metrics,
        "elapsed":       tlib.perf_counter() - t_start,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def integrate_linear(A, z0, t):
    """z(t_i) for dz/dt = A z on a uniform grid, via the matrix exponential."""
    step = expm(A * (t[1] - t[0]))
    Z = np.empty((A.shape[0], len(t)))
    Z[:, 0] = z0
    for k in range(1, len(t)):
        Z[:, k] = step @ Z[:, k - 1]
    return Z


def galerkin_maps(Phi):
    """(encoder, decoder) for an orthogonal POD projection (VC-H-OpInf)."""
    return Phi.T, Phi


def symplectic_maps(Phi, J):
    """(encoder, decoder) for the symplectic projection (Petrov-Galerkin, NiTROM)."""
    Jhat_inv = np.linalg.inv(Phi.T @ J @ Phi)
    return (J @ Phi).T, -Phi @ Jhat_inv


def rom_predict(encoder, decoder, A_dyn, x0, t):
    """Full-state ROM trajectory: decode the reduced flow from the encoded IC x0."""
    return decoder @ integrate_linear(A_dyn, encoder @ x0, t)


def global_relative_error(trajs, encoder, decoder, A_dyn, t):
    """Aggregate (Frobenius) relative error of a ROM over a set of FOM trajectories.

        e = sqrt( sum_j ||X_j - Xhat_j||_F^2 / sum_j ||X_j||_F^2 )

    X_j is the FOM reference (on grid t), Xhat_j the ROM prediction from X_j's IC.
    One number over ALL trajectories and snapshots -- the same metric for every
    model, so OpInf / PG / NiTROM are directly comparable.
    """
    num = den = 0.0
    for Xj in trajs:
        Xhat = rom_predict(encoder, decoder, A_dyn, Xj[:, 0], t)
        num += np.sum((Xj - Xhat) ** 2)
        den += np.sum(Xj ** 2)
    return float(np.sqrt(num / den))
