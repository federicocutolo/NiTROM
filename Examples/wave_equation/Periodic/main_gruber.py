"""
Reproduce the Gruber & Tezaur WaveEx.ipynb test case using the data-generation /
basis / output layout of main.py.

Physics (matches WaveEx.ipynb exactly):
  - domain [0, 1], PERIODIC boundary conditions (wrap-around Laplacian)
  - canonical Hamiltonian  x_dot = J A x,  A = blkdiag(-qxx, I),
    qxx = (c/dx)^2 * periodic second-difference   (so A == main.py's W)
  - wave speed c = 0.1
  - IC: q = h(s(x, a, 0.5)),  p = 0   (compact cubic-spline bump, ZERO momentum)
        s(x,a,b) = a|x-b|;  h(s) = 1 - 1.5 s^2 + 0.75 s^3   (s<=1)
                                   0.25 (2-s)^3             (1<s<=2)
                                   0                        (s>2)
  - 1-parameter family: a in [5, 15], 11 trajectories (npts = 11)

Outputs (into OUTPUT_SUBDIR; same filenames main.py writes):
  traj_%03d.npy, deriv_%03d.npy, weight_%03d.npy, time.npy, sample_params.npy,
  W_fom.npy, Phi_cl.npy  + diagnostic plots (ic / energy / snapshots / svd).
Point test_nitrom / train_nitrom at OUTPUT_SUBDIR to run the ROM pipeline on it.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))    # .../wave_equation/Periodic
WAVE = os.path.dirname(HERE)                          # .../wave_equation
if WAVE not in sys.path:
    sys.path.insert(0, WAVE)

from wave_foms import PeriodicWaveEquationFOM         # shared periodic FOM

# ── Config (WaveEx.ipynb values) ──────────────────────────────────────────────
N       = 500                          # spatial points per field (Nover2); state dim = 2N
L       = 1.0                          # domain length
c       = 0.1                          # wave speed
T       = 10.0                         # integration horizon
n_t     = 501                          # snapshots per trajectory
A_RANGE = np.linspace(5.0, 15.0, 11)   # IC steepness family (npts = 11)
CENTER  = 0.5                          # bump center
OUTPUT_SUBDIR = "trajectories"  # written next to this script, inside Periodic/


# ── Gruber-Tezaur initial condition: cubic-spline bump, zero momentum ─────────
def _s(x, a, b):
    return a * np.abs(x - b)


def _h(s):
    return np.where(s <= 1, 1.0 - 1.5 * s**2 + 0.75 * s**3,
                    np.where(s <= 2, 0.25 * (2.0 - s)**3, 0.0))


def gruber_ic(fom, a, center=0.5):
    q = _h(_s(fom.x, a, center))
    p = np.zeros_like(q)                          # zero initial momentum (standing pulse)
    return np.concatenate([q, p])


def generate_snapshots(fom, a_values, T, n_t, center=0.5):
    t = np.linspace(0.0, T, n_t)
    snaps, derivs, params = [], [], []
    for a in a_values:
        x0   = gruber_ic(fom, a, center)
        traj = fom.integrate(x0, t)
        snaps.append(traj)
        derivs.append(fom.rhs_matrix(traj))       # exact velocity snapshots
        params.append((a, center))
    return t, np.stack(snaps, 0), np.stack(derivs, 0), np.asarray(params, float)


if __name__ == "__main__":
    out_dir = os.path.join(HERE, OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    fom = PeriodicWaveEquationFOM(N=N, L=L, c=c)
    np.save(os.path.join(out_dir, "W_fom.npy"), fom.W)

    time, X_train, dX_train, params = generate_snapshots(fom, A_RANGE, T, n_t, CENTER)
    n_traj = X_train.shape[0]
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

    # ── Diagnostic plots ──────────────────────────────────────────────────────
    col = lambda k: plt.cm.viridis(k / max(n_traj - 1, 1))

    fig, ax = plt.subplots(figsize=(5.5, 4))            # (1) IC family
    for k in range(n_traj):
        ax.plot(fom.x, X_train[k, :fom.N, 0], color=col(k))
    ax.set_xlabel("x"); ax.set_ylabel("q(x, 0)")
    ax.set_title(f"Gruber-Tezaur ICs (a in [{A_RANGE[0]:.0f}, {A_RANGE[-1]:.0f}], p=0)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "ic.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))            # (2) Hamiltonian conservation
    for k in range(n_traj):
        H = fom.hamiltonian(X_train[k])
        ax.plot(time, H - H[0], color=col(k))
    ax.set_xlabel("t"); ax.set_ylabel(r"$H(t) - H(0)$")
    ax.set_title("Hamiltonian conservation (FOM)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "energy.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))              # (3) space-time of trajectory 0
    im = ax.pcolormesh(time, fom.x, X_train[0, :fom.N], shading="auto", cmap="RdBu_r")
    ax.set_xlabel("t"); ax.set_ylabel("x")
    ax.set_title(f"q(x, t), trajectory 0 (a={A_RANGE[0]:.0f})")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "snapshots.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))            # (4) singular-value decay
    ax.semilogy(1-explained, "o", ms=4)
    ax.axvline(var_rank, color="0.6", ls="--", label=f"var_rank = {var_rank}")
    ax.set_xlabel("mode"); ax.set_ylabel(r"$\sigma_i/\sigma_0$")
    ax.set_xlim(0, min(60, len(sqp))); ax.legend()
    ax.set_title("Cotangent-lift singular values")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "svd.png"), dpi=150); plt.close(fig)

    print(f"Saved data + basis + plots in {out_dir}")
