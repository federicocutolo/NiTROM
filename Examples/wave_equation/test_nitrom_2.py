"""
NiTROM vs. VC-H-OpInf equivalence check (with FOM reference).

For these settings the analytic NiTROM ROM and the VC-H-OpInf ROM should produce
the SAME full-space solution. We build both, compare them to the FOM, and plot the
OpInf-minus-NiTROM difference on its own axes to confirm it is at machine precision.

Testcase-agnostic: N is inferred from the basis and W_fom.npy is loaded from the data
dir, so it runs on the Dirichlet (main.py) and periodic (main_gruber.py) data alike.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Testcase selection + paths (registry lives in wave_cases.py) ──────────────
from wave_cases import get_case

TESTCASE  = "dirichlet"  # 'dirichlet' or 'periodic'
case      = get_case(TESTCASE)
train_dir = case.data_dir
out_dir   = case.output_dir
test_out  = case.test_results_dir
os.makedirs(out_dir,  exist_ok=True)
os.makedirs(test_out, exist_ok=True)


# ── VC-H-OpInf solver (Gruber & Tezaur, SIADS 2024, eq. 9) ────────────────────
def vc_h_opinf_solve(Phi, trajectories, derivatives):
    """
    Recover the symmetric reduced Hamiltonian operator A_bar by solving the Lyapunov
    system  G A_bar + A_bar G = M + M.T  with  Z = Phi.T X,  G = Z Z.T,  M = Phi.T J.T Xdot Z.T.
    Uses EXACT velocity snapshots Xdot (deriv_*.npy = J W X), not finite differences;
    H is quadratic so only A_bar is inferred.
    """
    r = Phi.shape[1]
    X     = np.concatenate(trajectories, axis=1)
    Xdot  = np.concatenate(derivatives,  axis=1)
    Z = Phi.T @ X
    M = (Phi.T @ J.T) @ Xdot @ Z.T
    G = Z @ Z.T
    LHS = np.kron(np.eye(r), G) + np.kron(G, np.eye(r))
    A_bar = np.linalg.solve(LHS, (M + M.T).flatten(order="F")).reshape(r, r, order="F")
    return 0.5 * (A_bar + A_bar.T)


# ── Load basis + FOM operator; build the reduced symplectic structure ─────────
Phi      = np.load(os.path.join(train_dir, "Phi_cl.npy"))   # (2N, r)
N        = Phi.shape[0] // 2
J        = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
Jhat     = Phi.T @ J @ Phi
Jhat_inv = np.linalg.inv(Jhat)
W_fom    = np.load(os.path.join(train_dir, "W_fom.npy"))    # H = 0.5 z^T W z
time     = np.load(os.path.join(train_dir, "time.npy"))
_x_path  = os.path.join(train_dir, "x.npy")                 # physical grid if saved, else node index
x_grid   = np.load(_x_path) if os.path.exists(_x_path) else np.arange(N)
print(f"Loaded basis from {train_dir}, shape: {Phi.shape}  (N={N})")

# ── Solve VC-H-OpInf -> A_bar ─────────────────────────────────────────────────
traj_files  = sorted(f for f in os.listdir(train_dir) if f.startswith("traj_")  and f.endswith(".npy"))
deriv_files = sorted(f for f in os.listdir(train_dir) if f.startswith("deriv_") and f.endswith(".npy"))
if len(deriv_files) != len(traj_files):
    raise FileNotFoundError(f"{len(traj_files)} traj_*.npy but {len(deriv_files)} deriv_*.npy in "
                            f"{train_dir}; rerun main.py to regenerate velocity snapshots.")
trajectories = [np.load(os.path.join(train_dir, f)) for f in traj_files]
derivatives  = [np.load(os.path.join(train_dir, f)) for f in deriv_files]
A_bar = vc_h_opinf_solve(Phi, trajectories, derivatives)

# ── Reduced dynamics  dz/dt = A z  for each ROM ───────────────────────────────
# VC-H-OpInf : POD coordinates,  z_dot = Jhat^{-T} A_bar z
A_dyn_opinf = Jhat_inv.T @ A_bar
# NiTROM (analytic, same A_bar): symplectic coordinates, z_dot = A_dyn_nit z.
# POD -> symplectic is a SIMILARITY transform (z_nit = -Jhat z_op), so the dynamics
# matrix transforms as A_dyn_nit = Jhat @ A_dyn_opinf @ Jhat^{-1} (= -A_bar @ Jhat^{-1}).
# Do NOT symmetrize: the dynamics is Jhat x (symmetric Hessian) with a purely imaginary
# spectrum; symmetrizing it forces real eigenvalues and the ROM blows up.
R = Jhat_inv.T @ A_bar @ Jhat_inv
A_dyn_nit   = Jhat @ R
print(f"max Re(lambda)  OpInf  = {np.linalg.eigvals(A_dyn_opinf).real.max():+.3e}"
      f"\nmax Re(lambda)  NiTROM = {np.linalg.eigvals(A_dyn_nit).real.max():+.3e}")

# ── Comparison setup ──────────────────────────────────────────────────────────
compare_idx = [0, len(trajectories) // 2]           # training trajectories 0 and 5
j_probe     = N // 2                                 # middle node (~ x = L/2)
decoder_op,  encoder_op  = Phi, Phi.T                       # OpInf  : POD projection
decoder_nit, encoder_nit = -Phi @ Jhat_inv, (J @ Phi).T    # NiTROM : symplectic projection


def integrate_linear(A, z0, t):
    """z(t_i) for dz/dt = A z on a uniform grid, via the matrix exponential."""
    step = expm(A * (t[1] - t[0]))
    Z = np.empty((A.shape[0], len(t)))
    Z[:, 0] = z0
    for k in range(1, len(t)):
        Z[:, k] = step @ Z[:, k - 1]
    return Z


# Re-integrate FOM + ROMs to T_SIM (data only covers [0, time[-1]]). The FOM is
# linear (x_dot = J W x), so the matrix exponential propagates it exactly.
T_SIM    = 8.0
dt       = time[1] - time[0]
time_sim = np.arange(0.0, T_SIM + 0.5 * dt, dt)
A_fom    = J @ W_fom

ns, ntraj = len(time_sim), len(trajectories)
alpha = np.array([float(np.load(os.path.join(train_dir, f"weight_{k:03d}.npy")))
                  for k in range(ntraj)])

# Full-space trajectories stored as 2N x (ntraj*ns); trajectory j -> columns [j*ns:(j+1)*ns].
X_fom = np.zeros((2*N, ntraj*ns))
X_op  = np.zeros_like(X_fom)
X_nit = np.zeros_like(X_fom)
for j in range(ntraj):
    x0 = trajectories[j][:, 0]
    sl = slice(j*ns, (j+1)*ns)
    X_fom[:, sl] = integrate_linear(A_fom,        x0,              time_sim)
    X_op[:,  sl] = decoder_op  @ integrate_linear(A_dyn_opinf, encoder_op  @ x0, time_sim)
    X_nit[:, sl] = decoder_nit @ integrate_linear(A_dyn_nit, encoder_nit @ x0, time_sim)

# Error matrix  e_j(t_i) = (1/alpha_j) ||X_fom_j(t_i) - Xhat_j(t_i)||^2   -> shape (ntraj, ns).
def error_matrix(Xhat):
    d = (X_fom - Xhat).reshape(2 * N, ntraj, ns)
    return np.sum(d ** 2, axis=0) / alpha[:, None]      # (ntraj, ns)

E_op  = error_matrix(X_op)
E_nit = error_matrix(X_nit)
# OpInf vs NiTROM difference (same weighted metric); ~machine precision if they coincide.
E_op_nit = np.sum((X_op - X_nit).reshape(2 * N, ntraj, ns) ** 2, axis=0) / alpha[:, None]

tab10  = plt.get_cmap("tab10").colors
COLORS = {"FOM": tab10[0], "OpInf": tab10[1], "NiTROM": tab10[3]}

# ── Static plot: q(L/2, t), FOM vs ROMs, on the compare_idx trajectories ──────
fig, axes = plt.subplots(1, len(compare_idx), figsize=(11, 4.2), sharex=True)
for ax, ti in zip(np.atleast_1d(axes), compare_idx):
    sl = slice(ti*ns, (ti+1)*ns)
    ax.plot(time_sim, X_fom[j_probe, sl], "-",  lw=2.0, color=COLORS["FOM"],    label="FOM")
    ax.plot(time_sim, X_op[j_probe, sl],  "--", lw=2.0, color=COLORS["OpInf"],  label="VC-H-OpInf")
    ax.plot(time_sim, X_nit[j_probe, sl], "-.", lw=1.6, color=COLORS["NiTROM"], label="NiTROM")
    ax.set_title(f"Training trajectory {ti}")
    ax.set_xlabel("time"); ax.set_ylabel(r"$q(L/2,\,t)$")
    ax.axhline(0, color="0.7", linewidth=0.5)
    fom_amp = np.max(np.abs(X_fom[j_probe, sl]))           # clip y to FOM range
    ax.set_ylim(-1.4 * fom_amp, 1.4 * fom_amp)
    if ti == compare_idx[0]:
        ax.legend(loc="upper right", framealpha=0.95)
plt.tight_layout()
out_png = os.path.join(test_out, "compare_fom_opinf_nitrom_TEST_2.png")
plt.savefig(out_png, dpi=150)
plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved {out_png}")

# ── Error vs time, per training trajectory ────────────────────────────────────
fig_e, ax_e = plt.subplots(figsize=(5, 4))
for traj in range(ntraj):
    ax_e.semilogy(time_sim, E_op[traj, :],  color=COLORS["OpInf"],  label="VC-H-OpInf" if traj == 0 else None)
    ax_e.semilogy(time_sim, E_nit[traj, :], color=COLORS["NiTROM"], label="NiTROM"     if traj == 0 else None)
ax_e.set_xlabel("time")
ax_e.set_ylabel(r"$\|x_j(t)-\hat{x}_j(t)\|^2 / \alpha_j$")
ax_e.set_title("Reconstruction error vs time for training trajectories")
ax_e.grid(True, which="both", alpha=0.3)
ax_e.legend(loc="upper right", framealpha=0.95)
fig_e.tight_layout()
err_png = os.path.join(test_out, "error_vs_time.png")
fig_e.savefig(err_png, dpi=150)
fig_e.savefig(err_png.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig_e)
print(f"Saved {err_png}")

# ── Trajectory-averaged error vs time ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.semilogy(time_sim, E_op.mean(axis=0),  color=COLORS["OpInf"],  label="VC-H-OpInf")
ax.semilogy(time_sim, E_nit.mean(axis=0), color=COLORS["NiTROM"], label="NiTROM")
ax.set_xlabel("time")
ax.set_ylabel(r"$\frac{1}{Ntraj} \ \sum_j \frac{1}{\alpha_j} \|x_j(t)-\hat{x}_j(t)\|^2$")
ax.set_title("Trajectories-averaged reconstruction error vs time")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", framealpha=0.95)
fig.tight_layout()
err_png = os.path.join(test_out, "error_vs_time_mean_TEST_2.png")
fig.savefig(err_png, dpi=150)
fig.savefig(err_png.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved {err_png}")

# ── OpInf vs NiTROM difference (own axes; machine precision if they coincide) ──
fig_d, ax_d = plt.subplots(figsize=(5, 4))
ax_d.semilogy(time_sim, E_op_nit.mean(axis=0), color="0.3", lw=1.5)
ax_d.axhline(np.finfo(float).eps, color="r", ls=":", lw=1.0, label="machine eps")
ax_d.set_xlabel("time")
ax_d.set_ylabel(r"$\frac{1}{Ntraj}\sum_j \frac{1}{\alpha_j}\|\hat{x}_j^{\rm OpInf}(t)-\hat{x}_j^{\rm NiTROM}(t)\|^2$")
ax_d.set_title("OpInf vs NiTROM solution difference")
ax_d.grid(True, which="both", alpha=0.3)
ax_d.legend(loc="best", framealpha=0.95)
fig_d.tight_layout()
diff_png = os.path.join(test_out, "diff_opinf_nitrom_TEST_2.png")
fig_d.savefig(diff_png, dpi=150)
fig_d.savefig(diff_png.replace(".png", ".pdf"), bbox_inches="tight")
plt.close(fig_d)
print(f"Saved {diff_png}")


# ── Videos: q(x, t) profile, FOM vs ROMs — one GIF per trajectory ─────────────
MAX_FRAMES    = 200                                # cap frames/GIF (decimate for speed/size)
VIDEO_SECONDS = 4.0                                # target playback duration
frame_ids     = np.arange(0, len(time_sim), max(1, len(time_sim) // MAX_FRAMES))
fps           = max(1.0, len(frame_ids) / VIDEO_SECONDS)

# Dirichlet: the state holds interior nodes only; pad the walls (u=0 at x=0 and x=L)
# so the boundary condition is visible. The periodic grid already spans [0, L].
if TESTCASE == "dirichlet":
    x_plot = np.concatenate(([0.0], x_grid, [x_grid[-1] + x_grid[0]]))
    def pad_bc(q):                          # (N, nt) -> (N+2, nt): zero rows at both walls
        z = np.zeros((1, q.shape[1]))
        return np.vstack([z, q, z])
else:
    x_plot = x_grid
    def pad_bc(q):
        return q


def make_traj_video(ti):
    x0 = trajectories[ti][:, 0]
    series = [("FOM",        "-",  1.0, integrate_linear(A_fom, x0, time_sim)[:N, :]),
              ("VC-H-OpInf", "--", 1.0,
               (decoder_op  @ integrate_linear(A_dyn_opinf, encoder_op  @ x0, time_sim))[:N, :]),
              ("NiTROM",     "-.", 1.0,
               (decoder_nit @ integrate_linear(A_dyn_nit,   encoder_nit @ x0, time_sim))[:N, :])]
    series = [(label, style, lw, pad_bc(q)) for label, style, lw, q in series]  # show BC walls

    fig_v, ax = plt.subplots(figsize=(6.5, 4.5))
    lines = [(ax.plot(x_plot, q[:, frame_ids[0]], style, lw=lw, label=label)[0], q)
             for label, style, lw, q in series]
    amp = 1.3 * np.max(np.abs(series[0][3])) + 1e-12   # FOM amplitude sets y-range
    ax.set_ylim(-amp, amp); ax.set_xlim(x_plot[0], x_plot[-1])
    ax.set_xlabel("x"); ax.set_ylabel(r"$q(x,\,t)$")
    ax.set_title(f"Training trajectory {ti}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    suptitle = fig_v.suptitle(f"t = {time_sim[frame_ids[0]]:.3f}")

    def _update(f):
        for ln, q in lines:
            ln.set_ydata(q[:, f])
        suptitle.set_text(f"t = {time_sim[f]:.3f}")
        return []

    anim = FuncAnimation(fig_v, _update, frames=frame_ids,
                         interval=1000.0 / fps, blit=False)
    gif_path = os.path.join(test_out, f"rom_vs_fom_traj_{ti:03d}_TEST_2.gif")
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig_v)
    return gif_path


for ti in range(len(trajectories)):
    print(f"Saved {make_traj_video(ti)}")
