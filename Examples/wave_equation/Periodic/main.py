"""
Gruber & Tezaur WaveEx test case (periodic BC), generated the same way as the
Dirichlet main.py: random Gaussian-pulse ICs sampled from a box (amplitude,
width, location). The single difference from the Dirichlet draw is the IC
regime, owned by wave_cases (sampling: momentum="zero", periodic=True):
  - ZERO initial momentum, so each displacement bump splits into TWO
    counter-propagating running waves (d'Alembert), and
  - a minimal-image (wrap-around) Gaussian, so any sampled location stays smooth
    across the periodic seam.

Physics:
  - domain [0, 1], PERIODIC boundary conditions (wrap-around Laplacian)
  - canonical Hamiltonian  x_dot = J A x,  A = blkdiag(-qxx, I),
    qxx = (c/dx)^2 * periodic second-difference   (so A == Dirichlet main.py's W)
  - wave speed c = 0.1

Outputs (into case.data_dir; same filenames main.py writes):
  traj_%03d.npy, deriv_%03d.npy, weight_%03d.npy, time.npy, sample_params.npy,
  W_fom.npy, Phi_cl.npy  + diagnostic plots (ic / energy / snapshots / svd).
Point test_nitrom / train_nitrom at the periodic case to run the ROM pipeline.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

HERE = os.path.dirname(os.path.abspath(__file__))    # .../wave_equation/Periodic
WAVE = os.path.dirname(HERE)                          # .../wave_equation
if WAVE not in sys.path:
    sys.path.insert(0, WAVE)

from wave_cases import get_case                       # physics (N, L, c) + paths + sampling box
import wave_sampling                                  # shared IC sampler (single source of truth)

# ── Config ────────────────────────────────────────────────────────────────────
n_traj = 10                            # number of trajectories to sample
n_t    = 100                           # snapshots per trajectory (including t=0)
T      = 10.0                          # integration horizon
SEED   = 42                            # training draw (test set uses a different seed)


if __name__ == "__main__":
    # Physics (N, L, c), the FOM, the sampling box, and I/O paths come from
    # wave_cases -> one source of truth, shared with the held-out test set in
    # test_nitrom.py (drawn from the same box, different seed).
    case    = get_case("periodic")
    fom     = case.fom
    ranges  = case.sample_ranges
    N, L, c = case.N, case.L, case.c

    out_dir = case.data_dir
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "W_fom.npy"), fom.W)

    time = np.linspace(0.0, T, n_t)
    X_train, dX_train, params = wave_sampling.generate_trajectories(
        fom, time, n_traj, seed=SEED, ranges=ranges)
    print(f"Generated {n_traj} trajectories | state dim {X_train.shape[1]} | "
          f"{X_train.shape[2]} snapshots | dx={fom.dx:.4e}")

    np.save(os.path.join(out_dir, "time.npy"),          time)
    np.save(os.path.join(out_dir, "x.npy"),             fom.x)
    np.save(os.path.join(out_dir, "sample_params.npy"), params)
    for k in range(n_traj):
        traj = X_train[k]
        w    = np.linalg.norm(fom.hamiltonian(traj))**0.5
        np.save(os.path.join(out_dir, f"traj_{k:03d}.npy"),   traj)
        np.save(os.path.join(out_dir, f"deriv_{k:03d}.npy"),  dX_train[k])
        np.save(os.path.join(out_dir, f"weight_{k:03d}.npy"), np.asarray([w]))

    # ── Cotangent-lift basis (same construction as main.py) → Phi_cl.npy ──────
    X_mat = X_train.transpose(1, 0, 2).reshape(2 * fom.N, -1)
    Q, P  = X_mat[:fom.N], X_mat[fom.N:]
    Uqp, sqp, _ = np.linalg.svd(np.hstack([Q, P]), full_matrices=False)
    explained = np.cumsum(sqp**2) / np.sum(sqp**2)
    var_rank  = int(np.argmax(explained > 0.99))
    Phi_qp = Uqp[:, :var_rank]
    zero   = np.zeros_like(Phi_qp)
    Phi_cl = np.block([[Phi_qp, zero], [zero, Phi_qp]])
    np.save(os.path.join(out_dir, "Phi_cl.npy"), Phi_cl)
    print(f"Cotangent-lift basis: spatial rank {var_rank}, shape {Phi_cl.shape}")

    # ── Diagnostic plots (Dirichlet main.py styling: grayscale, fontsize=14) ────
    gray = lambda k: plt.cm.gray(k / n_traj)            # match Dirichlet energy palette

    fig, ax = plt.subplots(figsize=(5.5, 4.4))            # (1) IC family
    for k in range(n_traj):
        ax.plot(fom.x, X_train[k, :fom.N, 0], color=gray(k))
    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$q(x,\,0)$", fontsize=14)
    ax.set_title(f"Initial displacement (two running waves)", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ic.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "ic.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))              # (2) Hamiltonian conservation
    for k in range(n_traj):
        H = fom.hamiltonian(X_train[k])
        ax.plot(time, H - H[0], color=gray(k), label=f"Traj {k+1}")
    ax.set_xlabel(r"$t$", fontsize=14)
    ax.set_ylabel(r"$H(t) - H(0)$", fontsize=14)
    ax.set_title("Hamiltonian conservation (FOM)", fontsize=14)
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "energy.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "energy.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))              # (3) space-time of trajectory 0
    im = ax.pcolormesh(time, fom.x, X_train[0, :fom.N], shading="auto", cmap="RdBu_r")
    ax.set_xlabel(r"$t$", fontsize=14); ax.set_ylabel(r"$x$", fontsize=14)
    ax.set_title("q(x, t), trajectory 0", fontsize=14)
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "snapshots.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "snapshots.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))              # (4) singular-value decay
    ax.semilogy(1 - explained, marker="o", color="k")
    ax.axvline(var_rank, color="k", linestyle="--", label=f"99% var at rank={var_rank}")
    ax.set_xlabel(r"Index $i$", fontsize=14)
    ax.set_ylabel(r"$ 1 - \frac{\left(\sum \sigma^2\right)_i}{\sum\sigma^2}$", fontsize=14)
    ax.set_xlim(0, min(60, len(sqp)))
    ax.legend(fontsize=12)
    ax.set_title("Explained variance", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "svd.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "svd.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved data + basis + plots in {out_dir}")
