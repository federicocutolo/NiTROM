"""
FOM vs. VC-H-OpInf vs. Petrov-Galerkin (vs. NiTROM) on training trajectories 0 and 5.
Loads the basis + training data, solves VC-H-OpInf in-script, assembles each ROM's
reduced dynamics, and writes a static comparison plot plus an animation.

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
    X    = np.concatenate(trajectories, axis=1)
    Xdot = np.concatenate(derivatives,  axis=1)
    Z = Phi.T @ X
    M = (Phi.T @ J.T) @ Xdot @ Z.T
    G = Z @ Z.T
    LHS = np.kron(np.eye(r), G) + np.kron(G, np.eye(r))
    A_bar = np.linalg.solve(LHS, (M + M.T).flatten(order="F")).reshape(r, r, order="F")
    return 0.5 * (A_bar + A_bar.T)


# ── Load basis + FOM operator; build the reduced symplectic structure ─────────
Phi      = np.load(os.path.join(train_dir, "Phi_cl.npy"))   # (2N, 2r)
N        = Phi.shape[0] // 2
J        = np.block([[np.zeros((N, N)), np.eye(N)], [-np.eye(N), np.zeros((N, N))]])
Jhat     = Phi.T @ J @ Phi
Jhat_inv = np.linalg.inv(Jhat)
W_fom    = np.load(os.path.join(train_dir, "W_fom.npy"))    # H = 0.5 z^T W z
W_hat    = Phi.T @ W_fom @ Phi
time     = np.load(os.path.join(train_dir, "time.npy"))
_x_path  = os.path.join(train_dir, "x.npy")                 # physical grid if saved, else node index
x_grid   = np.load(_x_path) if os.path.exists(_x_path) else np.arange(N)
print(f"Loaded basis from {train_dir}, shape: {Phi.shape}  (N={N})")

# ── Solve VC-H-OpInf -> A_bar; save for train_nitrom's warm start ─────────────
traj_files  = sorted(f for f in os.listdir(train_dir) if f.startswith("traj_")  and f.endswith(".npy"))
deriv_files = sorted(f for f in os.listdir(train_dir) if f.startswith("deriv_") and f.endswith(".npy"))
if len(deriv_files) != len(traj_files):
    raise FileNotFoundError(f"{len(traj_files)} traj_*.npy but {len(deriv_files)} deriv_*.npy in "
                            f"{train_dir}; rerun main.py to regenerate velocity snapshots.")
trajectories = [np.load(os.path.join(train_dir, f)) for f in traj_files]
derivatives  = [np.load(os.path.join(train_dir, f)) for f in deriv_files]
A_bar = vc_h_opinf_solve(Phi, trajectories, derivatives)
np.save(os.path.join(out_dir, "A_bar_opinf.npy"), A_bar)

# ── Reduced dynamics  dz/dt = A z  for each ROM ───────────────────────────────
A_dyn_opinf = Jhat_inv.T @ A_bar                    # VC-H-OpInf: z_dot = Jhat^{-T} A_bar z
# Petrov-Galerkin (intrusive H-ROM): z = (JU)^T x,  z_dot = E J grad H = U^T W x,
# closed with the decoder x = D z = -U Jhat^{-1} z  ->  z_dot = -W_hat Jhat^{-1} z.
# NOTE: grad H = W x (W_fom uses H = 1/2 x^T W x); a factor 2 here doubles the freq.
A_dyn_PG    = W_hat @ Jhat                           # (r, r)
print(f"max Re(lambda)  OpInf dynamics = {np.linalg.eigvals(A_dyn_opinf).real.max():+.3e}"
      f"\nmax Re(lambda) dynamics PG dynamics = {np.linalg.eigvals(A_dyn_PG).real.max():+.3e}")

# ── Trained NiTROM ROM (optional: Phi_nit, A2_nit from train_nitrom) ──────────
# Same symplectic projection as PG; H = z^T A2 z  =>  z_dot = Jhat_nit (A2 + A2^T) z.
nit_files = (os.path.join(out_dir, "Phi_nit.npy"), os.path.join(out_dir, "A2_nit.npy"))
have_nit  = all(os.path.exists(p) for p in nit_files)
if have_nit:
    Phi_nit, A2_nit = np.load(nit_files[0]), np.load(nit_files[1])
    Jhat_nit     = Phi_nit.T @ J @ Phi_nit
    Jhat_nit_inv = np.linalg.inv(Jhat_nit)
    A_dyn_nit    = Jhat_nit @ (A2_nit + A2_nit.T)
    decoder_nit, encoder_nit = -Phi_nit @ Jhat_nit_inv, (J @ Phi_nit).T
    print(f"max Re(lambda)  NiTROM dynamics= {np.linalg.eigvals(A_dyn_nit).real.max():+.3e}")
else:
    print("No Phi_nit.npy / A2_nit.npy in output -> NiTROM skipped (run train_nitrom.py).")

# ── Comparison setup ──────────────────────────────────────────────────────────
compare_idx = [0, len(trajectories) // 2]           # training trajectories 0 and 5
j_probe     = N // 2                                 # middle node (~ x = L/2)
decoder_op, encoder_op = Phi, Phi.T
decoder_pg, encoder_pg = -Phi @ Jhat_inv, (J @ Phi).T


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

# ── Static plot: q(L/2, t), FOM vs ROMs, on training trajectories 0 and 5 ─────
fig, axes = plt.subplots(1, len(compare_idx), figsize=(11, 4.2), sharex=True)
for ax, ti in zip(np.atleast_1d(axes), compare_idx):
    x0    = trajectories[ti][:, 0]
    q_fom = integrate_linear(A_fom, x0, time_sim)[j_probe, :]   # FOM re-integrated to T_SIM
    q_op  = (Phi        @ integrate_linear(A_dyn_opinf, encoder_op @ x0, time_sim))[j_probe, :]
    q_pg  = (decoder_pg @ integrate_linear(A_dyn_PG,    encoder_pg @ x0, time_sim))[j_probe, :]

    ax.plot(time_sim, q_fom, "-",  linewidth=2.0, label="FOM")
    ax.plot(time_sim, q_op,  "--", linewidth=1.4, label="VC-H-OpInf")
    ax.plot(time_sim, q_pg,  ":",  linewidth=1.6, label="Petrov-Galerkin")
    if have_nit:
        q_nit = (decoder_nit @ integrate_linear(A_dyn_nit, encoder_nit @ x0, time_sim))[j_probe, :]
        ax.plot(time_sim, q_nit, "-.", linewidth=1.6, label="NiTROM")

    ax.set_title(f"Training trajectory {ti}")
    ax.set_xlabel("time"); ax.set_ylabel(r"$q(L/2,\,t)$")
    ax.axhline(0, color="0.7", linewidth=0.5)
    fom_amp = np.max(np.abs(q_fom))                 # clip y to FOM range
    ax.set_ylim(-1.4 * fom_amp, 1.4 * fom_amp)
    if ti == compare_idx[0]:
        ax.legend(loc="upper right", framealpha=0.95)

plt.tight_layout()
out_png = os.path.join(test_out, "compare_fom_opinf_pg.png")
plt.savefig(out_png, dpi=150)
plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print(f"Saved {out_png}")


# ── Video: q(x, t) profile, FOM vs ROMs, over the same two trajectories ────────
# FOM re-integrated to T_SIM so the pulse reflects off the walls; the ROMs
# extrapolate past the training horizon. Same B&W styles as the static plot.
panels = []
for ti in compare_idx:
    x0 = trajectories[ti][:, 0]
    # (label, style, linewidth, q(x,t)); FOM first so its amplitude sets the view.
    series = [("FOM",             "-",   2.0, integrate_linear(A_fom, x0, time_sim)[:N, :]),
              ("VC-H-OpInf",      "--",  1.4,
               (Phi        @ integrate_linear(A_dyn_opinf, encoder_op @ x0, time_sim))[:N, :]),
              ("Petrov-Galerkin", ":",   1.6,
               (decoder_pg @ integrate_linear(A_dyn_PG,    encoder_pg @ x0, time_sim))[:N, :])]
    # if have_nit:
    #     series.append(("NiTROM", "k-.", 1.6,
    #                    (decoder_nit @ integrate_linear(A_dyn_nit, encoder_nit @ x0, time_sim))[:N, :]))
    panels.append((ti, series))

fig_v, axes_v = plt.subplots(1, 2, figsize=(12, 4.5))
panel_lines = []                                   # per panel: list of (line, q(x,t))
for ax, (ti, series) in zip(np.atleast_1d(axes_v), panels):
    lines = [(ax.plot(x_grid, q[:, 0], style, lw=lw, label=label)[0], q)
             for label, style, lw, q in series]
    amp = 1.3 * np.max(np.abs(series[0][3])) + 1e-12   # FOM amplitude sets y-range
    ax.set_ylim(-amp, amp); ax.set_xlim(x_grid[0], x_grid[-1])
    ax.set_xlabel("x"); ax.set_ylabel(r"$q(x,\,t)$")
    ax.set_title(f"Training trajectory {ti}")
    panel_lines.append(lines)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
suptitle = fig_v.suptitle(f"t = {time_sim[0]:.3f}")


def _update(f):
    for lines in panel_lines:
        for ln, q in lines:
            ln.set_ydata(q[:, f])
    suptitle.set_text(f"t = {time_sim[f]:.3f}")
    return []


video_seconds = 4.0                                # target playback duration
fps           = max(1.0, len(time_sim) / video_seconds)
anim = FuncAnimation(fig_v, _update, frames=len(time_sim), interval=1000.0 / fps, blit=False)
gif_path = os.path.join(test_out, "rom_vs_fom_training.gif")
anim.save(gif_path, writer=PillowWriter(fps=fps))
plt.close(fig_v)
print(f"Saved {gif_path}")
