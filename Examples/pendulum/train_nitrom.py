import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from mpi4py import MPI


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers

from NiTROM.Optimization_Functions import classes, nitrom_functions
from NiTROM.PyManopt_Functions.my_pymanopt_classes import myAdaptiveLineSearcher
from NiTROM.Optimization_Functions.classes import HamiltonianROM
from pendulum_utils import SRKN_REFERENCE_STEP, g, hamiltonian, l, m, pdot, qdot, srkn_pendulum_b6


DATA_DIR = os.path.join(HERE, "trajectories")


class PendulumFOM:
    def hamiltonian(self, z):
        return hamiltonian(z)

    def qdot(self, p):
        return qdot(p)

    def pdot(self, q):
        return pdot(q)

    def integrate(self, z0, time):
        return srkn_pendulum_b6(
            x0=z0,
            t0=time[0],
            tf=time[-1],
            h=SRKN_REFERENCE_STEP,
            t_eval=time,
        )[0]


def initial_tensors():
    # A2 = np.array([
    #     [0.5 * m * g * l, 0.0],
    #     [0.0, 1.0 / (2.0 * m * l**2)],
    # ])
    A2 = np.zeros((2,2))
    A3 = np.zeros((2, 2, 2))
    A4 = np.zeros((2, 2, 2, 2))
    A4[0, 0, 0, 0] = -(m * g * l) / 24.0
    return A2, A3, A4


def integrate_hamiltonian_rom(rom, z0_full, time):
    ops = rom.build_projection_operators()
    z0_rom = ops["encoder"] @ z0_full
    dt = SRKN_REFERENCE_STEP
    Z_rom = rom.integrate(time, z0_rom, dt=dt)
    X_rom = ops["decoder"] @ Z_rom
    H_rom = np.array([rom.compute_hamiltonian(Z_rom[:, i]) for i in range(Z_rom.shape[1])])
    return X_rom, H_rom, Z_rom


def evaluate_rom_on_test_case(rom, fom, z0, time):
    X_fom = fom.integrate(z0, time)
    X_rom, H_rom, Z_rom = integrate_hamiltonian_rom(rom, z0, time)
    H_fom = np.array([fom.hamiltonian(X_fom[:, i]) for i in range(X_fom.shape[1])])
    return X_fom, H_fom, X_rom, H_rom, Z_rom


def main():
    fname_traj = os.path.join(DATA_DIR, "traj_%03d.npy")
    fname_weight = os.path.join(DATA_DIR, "weight_%03d.npy")
    fname_time = os.path.join(DATA_DIR, "time.npy")

    traj_files = sorted(
        fname for fname in os.listdir(DATA_DIR)
        if fname.startswith("traj_") and fname.endswith(".npy")
    )
    n_traj = len(traj_files)
    if n_traj == 0:
        raise ValueError("No trajectory files found. Run datagen.py first.")
    print(f"Found {n_traj} trajectory files. Using these for training.")

    pool = classes.mpi_pool(
        MPI.COMM_WORLD,
        n_traj,
        fname_traj,
        fname_time,
        fname_weights=fname_weight,
    )

    fom = PendulumFOM()
    r = 2
    poly_comp = [1, 2, 3]

    which_trajs = np.arange(pool.my_n_traj)
    which_times = np.arange(pool.n_snapshots)
    opt_obj = classes.optimization_objects(
        pool,
        which_trajs,
        which_times,
        leggauss_deg=5,
        nsave_rom=5,
        poly_comp=poly_comp,
        hamiltonian=True,
    )

    St = manifolds.Stiefel(2, r)
    Euc_rr = manifolds.Euclidean(r, r)
    Euc_rrr = manifolds.Euclidean(r, r, r)
    Euc_rrrr = manifolds.Euclidean(r, r, r, r)
    M = manifolds.Product([St, St, Euc_rr, Euc_rrr, Euc_rrrr])

    cost, grad, _ = nitrom_functions.create_objective_and_gradient(M, opt_obj, pool, fom)
    problem = pymanopt.Problem(M, cost, euclidean_gradient=grad)

    Phi0 = np.eye(2)
    Psi0 = np.eye(2)
    A20, A30, A40 = initial_tensors()
    point0 = (Phi0, Psi0, A20, A30, A40)

    initial_cost = cost(*point0)
    grad0 = grad(*point0)
    if pool.rank == 0:
        print("Initial cost:", initial_cost)
        grad_norm = sum(np.linalg.norm(g) for g in grad0)
        print("Initial summed gradient norm:", grad_norm)

    line_searcher = myAdaptiveLineSearcher(
        contraction_factor=0.5,
        sufficient_decrease=0.5,
        max_iterations=25,
        initial_step_size=1.0,
    )
    optimizer = optimizers.ConjugateGradient(
        max_iterations=10,
        min_step_size=1.0e-20,
        max_time=3600,
        line_searcher=line_searcher,
        log_verbosity=1,
        verbosity=2 if pool.rank == 0 else 0,
    )

    result = optimizer.run(problem, initial_point=point0)

    Phi_opt = result.point[0]
    Psi_opt = result.point[1]
    Phi_opt = Phi_opt @ scipy.linalg.inv(Psi_opt.T @ Phi_opt)
    tensors_opt = tuple(result.point[2:])

    rom = classes.HamiltonianROM(
        operators=list(tensors_opt),
        poly_comp=poly_comp,
        Phi=Phi_opt,
    )

    test_time = np.arange(0.0, 100.0 + 0.5 * SRKN_REFERENCE_STEP, SRKN_REFERENCE_STEP)
    z0_test = np.array([np.pi/4, 0.0])
    X_fom, H_fom, X_rom, H_rom, Z_rom = evaluate_rom_on_test_case(rom, fom, z0_test, test_time)
    final_cost = cost(result.point[0], result.point[1], *result.point[2:])

    if pool.rank == 0:
        print("Final cost:", final_cost)
        print("Optimized Phi:\n", Phi_opt)
        print("Optimized A2:\n", tensors_opt[0])
        print("Optimized A3:\n", tensors_opt[1])
        print("Optimized A4[0,0,0,0]:", tensors_opt[2][0, 0, 0, 0])
        print("Test max |q_fom - q_rom|:", np.max(np.abs(X_fom[0] - X_rom[0])))
        print("Test max |p_fom - p_rom|:", np.max(np.abs(X_fom[1] - X_rom[1])))
        print("Test max FOM energy drift:", np.max(np.abs(H_fom - H_fom[0])))
        print("Test max ROM energy drift:", np.max(np.abs(H_rom - H_rom[0])))

        np.save(os.path.join(HERE, "Phi_nit.npy"), Phi_opt)
        np.save(os.path.join(HERE, "Psi_nit.npy"), Psi_opt)
        np.save(os.path.join(HERE, "A2_nit.npy"), tensors_opt[0])
        np.save(os.path.join(HERE, "A3_nit.npy"), tensors_opt[1].reshape((r, r**2)))
        np.save(os.path.join(HERE, "A4_nit.npy"), tensors_opt[2].reshape((r, r**3)))

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(X_fom[0], X_fom[1], label="FOM")
        plt.plot(X_rom[0], X_rom[1], "--", label="Hamiltonian ROM")
        plt.plot(Z_rom[0], Z_rom[1], ":", label="ROM in latent space")
        plt.xlabel("q")
        plt.ylabel("p")
        plt.title("Pendulum Phase Space")
        # place legend  bottom left
        plt.legend(loc="lower left")

        plt.subplot(1, 2, 2)
        plt.plot(test_time, np.abs(H_fom - H_fom[0]), label="FOM")
        plt.plot(test_time, np.abs(H_rom - H_rom[0]), "--", label="Hamiltonian ROM")
        plt.xlabel("t")
        plt.ylabel("|H(t) - H(0)|")
        plt.yscale("log")
        plt.title("Pendulum Energy Drift")
        plt.legend()

        plt.tight_layout()
        fig_path = os.path.join(HERE, "trained_pendulum_rom.pdf")
        plt.savefig(fig_path, bbox_inches="tight")
        print(f"Saved figure to {fig_path}")
        plt.show()


if __name__ == "__main__":
    main()
