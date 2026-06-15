import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
import seaborn as sns

sns.set_style("ticks")

HERE = os.path.dirname(os.path.abspath(__file__))    # .../wave_equation/Dirichlet
WAVE = os.path.dirname(HERE)                          # .../wave_equation
if WAVE not in sys.path:
    sys.path.insert(0, WAVE)

from wave_foms import WaveEquationFOM                 # shared Dirichlet FOM

# ROM basis: main.py writes Phi_cl.npy (cotangent lift) or Phi_pod.npy (plain POD).
# Downstream scripts load the matching file directly.
USE_COTANGENT_LIFT = True   # True -> symplectic cotangent-lift basis; False -> plain POD

# ── Data generation ───────────────────────────────────────────────────────────

def gaussian_ic(fom, amp, width, x0, direction):
    x = fom.x
    q = amp * np.exp(-(x - x0)**2 / (2 * width**2))
    # p = direction * fom.c * np.gradient(q, fom.dx)
    p = direction * fom.c * ((x-x0)/width**2) * q
    

    return np.concatenate([q, p])

SAMPLE_RANGES = {
    "amp":   (0.1, 1.0),
    "sigma": None,           # set inside generate_snapshots once fom.dx is known
    "x0":    (0.5, 1.5),
}

def generate_snapshots(fom, n_traj, T, n_t):
    seed = 42
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, T, n_t)
    # sigma floor as a multiple of dx (resolution); dx*5 keeps the range valid at
    # L=2 (dx*10 = 0.155 would exceed the 0.15 ceiling) and matches the old L=1 widths.
    SAMPLE_RANGES["sigma"] = (fom.dx*5, 0.15)
    snaps, derivs, params = [], [], []
    for _ in range(n_traj):
        amp       = rng.uniform(*SAMPLE_RANGES["amp"])
        sigma     = rng.uniform(*SAMPLE_RANGES["sigma"])
        mu        = rng.uniform(*SAMPLE_RANGES["x0"])
        direction = rng.choice([-1, 1])
        x0        = gaussian_ic(fom, amp=amp, width=sigma, x0=mu, direction=direction)
        traj      = fom.integrate(x0, t)
        snaps.append(traj)
        derivs.append(fom.rhs_matrix(traj))     # exact dz/dt snapshots
        params.append((amp, sigma, mu, direction))
    return (t, np.stack(snaps, axis=0), np.stack(derivs, axis=0),
            np.asarray(params, dtype=float))


def plot_sampling(fom, params, X_train, save_path):
    """Save two separate figures: parameter scatter and overlaid q(x,0)."""
    A, sigma = params[:, 0], params[:, 1]
    base, ext = os.path.splitext(save_path)

    # Figure 1: parameter-space scatter ----------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    A_lo, A_hi = SAMPLE_RANGES["amp"]
    s_lo, s_hi = SAMPLE_RANGES["sigma"]
    ax.add_patch(plt.Rectangle((A_lo, s_lo), A_hi - A_lo, s_hi - s_lo,
                               facecolor="none", edgecolor="k",
                               linestyle="--", linewidth=1.0, zorder=0,
                               label="sampling box"))
    ax.scatter(A, sigma, facecolor="k", edgecolor="k", s=60, marker="o")
    ax.set_xlabel(r"wave amplitude $A$", fontsize=14)
    ax.set_ylabel(r"Gaussian width $\sigma$", fontsize=14)
    ax.set_title("Sampling", fontsize=14)
    ax.margins(x=0.1, y=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sampling.png"))
    plt.savefig(os.path.join(output_dir, f"sampling.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Figure 2: resulting initial conditions -----------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    x_plot = np.concatenate(([0.0], fom.x, [fom.L]))
    for k in range(len(params)):
        q0    = X_train[k, :fom.N, 0]
        q_pad = np.concatenate(([0.0], q0, [0.0]))
        ax.plot(x_plot, q_pad, color="k", linewidth=1.2, alpha=0.85)
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$q(x,\,0)$", fontsize=14)
    ax.set_title(r"Initial displacement $q(x,0)$", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ic.png"))
    plt.savefig(os.path.join(output_dir, f"ic.pdf"), bbox_inches="tight")
    plt.close(fig)


# Generate movies of trajectories
def make_trajectory_movie(fom, time, X_train, output_path, seed):
    rng = np.random.default_rng(seed)
    traj_idx = rng.integers(X_train.shape[0])
    frame_ids = np.arange(0, len(time), max(1, len(time) // 200))
    q = X_train[traj_idx, :fom.N]
    q_min = q[:, frame_ids].min()
    q_max = q[:, frame_ids].max()

    fig, ax = plt.subplots(figsize=(5, 4))
    x_plot = np.concatenate(([0.0], fom.x, [fom.L]))
    q_pad  = np.pad(q, ((1, 1), (0, 0)))   # zeros top & bottom rows
    line,  = ax.plot(x_plot, q_pad[:, 0], color="k")

    # ax.set_xlim(fom.x[0], fom.x[-1])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("q(x,t)")
    title = ax.set_title(f"trajectory {traj_idx}, t = {time[0]:.3f}")

    def update(frame):
        k = frame_ids[frame]
        line.set_ydata(q_pad[:, k])
        title.set_text(f"trajectory {traj_idx}, t = {time[k]:.3f}")
        return [line, title]

    anim = FuncAnimation(fig, update, frames=len(frame_ids), blit=True)
    anim.save(output_path, writer=PillowWriter(fps=20))
    plt.close(fig)


# Generate training objects for NiTROM optimization
if __name__ == "__main__":
    N = 128             # number of spatial grid points (state dimension n=2N)
    L = 2.0             # domain length: wide enough that the Gaussian ICs decay to
                        # ~0 before the Dirichlet walls (x0<=0.8, 4*sigma<=0.6)
    c = 1.0             # wave speed
    n_traj = 10         # number of trajectories to generate
    n_t = 100           # number of time points per trajectory (including t=0)
    output_dir = "trajectories"   # written next to this script, inside Dirichlet/

    T = 2.0             # integration time for each trajectory
    # Define FOM
    fom = WaveEquationFOM(N=N, L=L, c=c)  

    # Genrate and save trajectories 
    # Erase trajectory folder if it exists
    traj_dir = os.path.join(HERE, output_dir)
    # if os.path.exists(traj_dir):
    #     for fname in os.listdir(traj_dir):
    #         os.remove(os.path.join(traj_dir, fname))

    np.save(os.path.join(traj_dir, "W_fom.npy"), fom.W)

    time, X_train, dX_train, sample_params = generate_snapshots(fom, n_traj=n_traj, T=T, n_t=n_t)
    print(X_train.shape)  # should be (n_traj, n_states, n_t)

    output_dir = os.path.join(HERE, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "time.npy"), time)
    np.save(os.path.join(output_dir, "x.npy"), fom.x)
    np.save(os.path.join(output_dir, "sample_params.npy"), sample_params)

    plot_sampling(fom, sample_params, X_train,
                  os.path.join(output_dir, "sampling.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    fig, ax2 = plt.subplots(figsize=(6, 4))

    for k in range(n_traj):
        traj = X_train[k]
        traj_energy = fom.hamiltonian(traj)
        traj_weight = np.linalg.norm(traj_energy)**0.5
        # print(f"weight_{k:03d}:", traj_weight)
        
        np.save(os.path.join(output_dir, f"traj_{k:03d}.npy"), traj)
        np.save(os.path.join(output_dir, f"deriv_{k:03d}.npy"), dX_train[k])
        np.save(os.path.join(output_dir, f"weight_{k:03d}.npy"), np.asarray([traj_weight]))
        
        normalized_energy = traj_energy / np.mean(traj_energy)
        # Set black and white gradient for trajectories 
        ax.semilogy(time, normalized_energy, color=plt.cm.gray(k/n_traj), label=f"Traj {k+1}")
        ax.set_xlabel("time", fontsize=14)
        ax.set_ylabel(r"$\frac{\mathcal{H}_{Traj}}{\sqrt{\|\mathcal{H}_{Traj}\|_2}}$", fontsize=14)
        ax.set_title(f"Trajectory-normalized Energy")
        ax.legend(fontsize=10, loc="upper left", ncol=2)

        # Plot initial coditions
        q0 = traj[:fom.N, 0]
        p0 = traj[fom.N:, 0]
        ax2.plot(fom.x, q0, color='k', label="q0",)
        ax2.plot(fom.x, p0, linestyle='dashed', color='k', label="p0",)
        if k == 1:
            ax2.legend(fontsize=10)

    ax2.set_xlabel("x", fontsize=14)
    ax2.set_ylabel("Initial condition", fontsize=14)
    ax2.set_title(f"Initial conditions for {n_traj} trajectories")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"initial_conditions.png"))
    plt.savefig(os.path.join(output_dir, f"initial_conditions.pdf"), bbox_inches="tight")
    plt.close()

    plt.savefig(os.path.join(output_dir, f"energy.png"))
    plt.savefig(os.path.join(output_dir, f"energy.pdf"), bbox_inches="tight")
    print(f"Saved {n_traj} trajectories in {output_dir}")
    
    make_trajectory_movie(fom, time, X_train, os.path.join(output_dir, "movie_traj_000.gif"), seed = 42)
    make_trajectory_movie(fom, time, X_train, os.path.join(output_dir, "movie_traj_001.gif"), seed = 1)
    print(f"Saved trajectory movies in {output_dir}")

    # Snapshot matrix: (2N, Ntraj*Nsnaps); column i*Nsnaps+j is x[i, :, j]
    X_train_matrix = X_train.transpose(1, 0, 2).reshape(X_train.shape[1], -1)
    print(f"Snapshot matrix shape: {X_train_matrix.shape} (n_states, n_traj*n_snapshots)")

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range (10):
        ax.plot(fom.x, X_train_matrix[:fom.N, i])
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("Snapshot", fontsize=14)
    # ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"snapshots.png"))

    # ── ROM basis -> Phi_cl.npy / Phi_pod.npy (USE_COTANGENT_LIFT selects how) ─
    if USE_COTANGENT_LIFT:
        # Cotangent lift (Peng & Mohseni 2016): POD of the combined [Q, P] matrix
        # treated as N-dim "spatial" snapshots, then diagonal-stacked. The same
        # spatial basis is used for both the q-block and the p-block, so Phi.T J Phi
        # is canonical [[0, I],[-I, 0]] -- cond(Jhat) = 1 by construction.
        Q = X_train_matrix[:fom.N, :]
        P = X_train_matrix[fom.N:, :]
        X_qp = np.hstack([Q, P]) 
        print("[shape check] X_qp:", X_qp.shape)                               # (N, 2*n_s)
        Uqp, sqp, _ = np.linalg.svd(X_qp, full_matrices=False)
        print("[shape check] Uqp:", Uqp.shape)                               # (N, 2*n_s)

        explained_var = np.cumsum(sqp**2) / np.sum(sqp**2)
        var_rank = int(np.argmax(explained_var > 0.99))   # rank to capture 99% variance in [Q; P]

        Phi_qp  = Uqp[:, :var_rank]
        print("[shape check] Phi_qp:", Phi_qp.shape)                               # (N, 2*n_s)

        zero    = np.zeros_like(Phi_qp)
        Phi_cl = np.block([[Phi_qp, zero],
                            [zero,   Phi_qp]]) # shape (2N, 2*var_rank)
        print("[shape check] Phi_cl:", Phi_cl.shape)                               # (N, 2*n_s)

        captured_variance = explained_var[var_rank]
        print(f"Cotangent-lift basis (Peng-Mohseni): spatial rank={var_rank} "
              f"(captures {captured_variance*100:.4f}% energy), "
              f"basis shape={Phi_cl.shape}")

    else:
        # Plain POD of the full (q, p) state.
        U, s, _ = np.linalg.svd(X_train_matrix, full_matrices=False)
        explained_var = np.cumsum(s**2) / np.sum(s**2)
        var_rank = int(np.argmax(explained_var > 0.99))
        if var_rank % 2:
            var_rank += 1
        Phi_pod = U[:, :var_rank]
        print(f"Plain POD basis: r={Phi_pod.shape[1]}")

    basis      = Phi_cl if USE_COTANGENT_LIFT else Phi_pod
    basis_file = "Phi_cl.npy" if USE_COTANGENT_LIFT else "Phi_pod.npy"
    np.save(os.path.join(output_dir, basis_file), basis)
    print(f"Saved {basis_file} (cotangent_lift={USE_COTANGENT_LIFT}) in {output_dir}")


    # Plot
    ax, fig = plt.subplots(figsize=(6, 5))
    plt.semilogy((1 - explained_var), marker="o", color="k")
    plt.xlabel(r"Index $i$", fontsize=14)
    plt.ylabel(r"$ 1 - \frac{\left(\sum \sigma^2\right)_i}{\sum\sigma^2}$", fontsize=14)
    plt.axvline(var_rank, color="k", linestyle="--", label=f"99% var at rank={var_rank}")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.title("Explained variance", fontsize=14)
    plt.savefig(os.path.join(output_dir, f"svd.png"))
    plt.savefig(os.path.join(output_dir, f"svd.pdf"), bbox_inches="tight")

    plt.plot()

    # plot2
    fig, ax = plt.subplots(figsize=(6, 5))

    q0 = Phi_cl[:fom.N, 0]
    p0 = Phi_cl[fom.N:, 0]
    ax.plot(fom.x, q0, color='k', label="q",)
    ax.plot(fom.x, p0, linestyle='dashed', color='k', label="p",)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("POD mode", fontsize=14)
    plt.savefig(os.path.join(output_dir, f"pod_mode_0.png"))
    plt.savefig(os.path.join(output_dir, f"pod_mode_0.pdf"), bbox_inches="tight")

