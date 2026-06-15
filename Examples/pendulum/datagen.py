import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "trajectories")

from pendulum_utils import hamiltonian, srkn_pendulum_b6
    
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    # Cleanup trajectories directory
    for name in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    time = np.linspace(0.0, 20.0, 401)
    np.save(os.path.join(DATA_DIR, "time.npy"), time)

    initial_conditions = [
        np.array([0.10, 0.00]),
        np.array([0.20, 0.00]),
    ]

    print(f"Generating {len(initial_conditions)} pendulum trajectories in {DATA_DIR}")
    for k, x0 in enumerate(initial_conditions):
        traj, H = srkn_pendulum_b6(
            x0=x0,
            t0=time[0],
            tf=time[-1],
            h=(time[1] - time[0]) / 2,
            t_eval=time,
        )
        
        weight = np.array([max(hamiltonian(x0), 1.0e-8)])

        np.save(os.path.join(DATA_DIR, f"traj_{k:03d}.npy"), traj)
        np.save(os.path.join(DATA_DIR, f"weight_{k:03d}.npy"), weight)

        print(
            f"  saved traj_{k:03d}.npy with z0={x0}, "
            f"max |q|={np.max(np.abs(traj[0])):.4f}, "
            f"max |p|={np.max(np.abs(traj[1])):.4f}"
        ) 

    print("Done.")

    # Plots
    plt.figure(figsize=(14, 7))
    plt.tight_layout()
    plt.suptitle("Symplectic schemes", fontsize=14)

    plt.subplot(1,2,1)
    # plt.plot(q_e, p_e, linestyle='dashed', label='Euler')
    plt.plot(traj[0], traj[1], label='Symplectic RK4')
    plt.title("Phase space")
    plt.xlim([-1.0, 1.0])
    plt.xlabel(r'$\theta$'); plt.ylabel('p')
    plt.legend()

    plt.subplot(1,2,2)
    # plt.plot(t, np.abs(H_e-H_e[0]), linestyle='dashed', label='Euler')
  
    plt.plot(time, np.abs(H-H[0]), label='SRNK')
    plt.yscale('log')
    plt.title("Energy")
    plt.xlabel("t")
    plt.ylabel("H-H0")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
