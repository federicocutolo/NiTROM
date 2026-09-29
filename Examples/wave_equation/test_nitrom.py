"""
Evaluate the ROMs (VC-H-OpInf, Petrov-Galerkin, NiTROM) against the FOM on TWO
datasets:
  * the TRAINING trajectories (data the basis/operators were fit on), and
  * a HELD-OUT TEST set: N_TEST new ICs drawn from the same sampling space as
    training (wave_sampling.py) but with a different seed, so the ROMs are judged
    on data they never saw.
The reduced operators are always built from the TRAINING data; only the
evaluation ICs differ between the two passes.

For each dataset it writes a static q(L/2, t) comparison, reconstruction-error
curves, and one q(x, t) animation per trajectory, with a '_train' / '_test'
filename suffix. The NiTROM training history is written once (dataset-agnostic).

Testcase-agnostic: N is inferred from the basis and the FOM/W_fom come from
wave_cases, so it runs on the Dirichlet and periodic data alike.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
sns.set(style="ticks", font_scale=1.2, palette="tab10")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Testcase selection + paths (registry lives in wave_cases.py) ──────────────
from wave_cases import get_case, nitrom_output_dir
import wave_roms
import wave_sampling

TESTCASE  = "dirichlet"  # 'dirichlet' or 'periodic'
BASIS     = "cl"         # reduced basis used by ALL models: "cl" | "pod"
BASIS_TAG = {"cl": "CL basis", "pod": "POD basis"}[BASIS]  # figure-title tag
SHOW_NITROM_PG = False   # plot the PG-initialized NiTROM too (OpInf is the better
                         # init here, so only the OpInf-initialized NiTROM is shown).
                         # The PG and OpInf *baselines* are always plotted.
case      = get_case(TESTCASE)
train_dir = case.data_dir
test_out  = os.path.join(case.test_results_dir, BASIS)   # plots for THIS basis
os.makedirs(test_out, exist_ok=True)
train_out    = os.path.join(test_out, "training")   # comparisons on the training trajectories
test_set_out = os.path.join(test_out, "testing")    # comparisons on the held-out test trajectories
os.makedirs(train_out,    exist_ok=True)
os.makedirs(test_set_out, exist_ok=True)


# ── Load basis + FOM operator; build the reduced symplectic structure ─────────
Phi_cl      = np.load(os.path.join(train_dir, "Phi_cl.npy"))   # (2N, 2r)
Phi_pod     = np.load(os.path.join(train_dir, "Phi_pod.npy"))
Phi_rand    = np.random.randn(*Phi_cl.shape) - 1e7 * np.block([
                                                            [np.zeros((Phi_cl.shape[0]//2, Phi_cl.shape[1]//2)), np.ones((Phi_cl.shape[0]//2, Phi_cl.shape[1]//2))],
                                                            [np.ones((Phi_cl.shape[0]//2, Phi_cl.shape[1]//2)), np.zeros((Phi_cl.shape[0]//2, Phi_cl.shape[1]//2))]])  # random, but not symplectic  # random, but not symplectic
Phi = {"cl": Phi_cl, "pod": Phi_pod}[BASIS]

N        = Phi.shape[0] // 2
if N != case.N:
    raise ValueError(
        f"Stale data: basis/trajectories in {train_dir} have N={N} but the "
        f"'{TESTCASE}' case is configured with N={case.N} (wave_cases.py). The "
        f"generated test set would not match -- rerun main.py to regenerate "
        f"{train_dir} with the current physics.")
J        = wave_roms.canonical_J(2 * N)
Jhat     = Phi.T @ J @ Phi
Jhat_inv = np.linalg.inv(Jhat)
W_fom    = np.load(os.path.join(train_dir, "W_fom.npy"))    # H = 0.5 z^T W z
time     = np.load(os.path.join(train_dir, "time.npy"))
_x_path  = os.path.join(train_dir, "x.npy")                 # physical grid if saved, else node index
x_grid   = np.load(_x_path) if os.path.exists(_x_path) else np.arange(N)
print(f"Loaded basis from {train_dir}, shape: {Phi.shape}  (N={N})")

# ── Baseline ROM operators: built from the TRAINING data (never the test set) ─
traj_files  = sorted(f for f in os.listdir(train_dir) if f.startswith("traj_")  and f.endswith(".npy"))
deriv_files = sorted(f for f in os.listdir(train_dir) if f.startswith("deriv_") and f.endswith(".npy"))
if len(deriv_files) != len(traj_files):
    raise FileNotFoundError(f"{len(traj_files)} traj_*.npy but {len(deriv_files)} deriv_*.npy in "
                            f"{train_dir}; rerun main.py to regenerate velocity snapshots.")
train_trajs  = [np.load(os.path.join(train_dir, f)) for f in traj_files]
train_derivs = [np.load(os.path.join(train_dir, f)) for f in deriv_files]
A_bar        = wave_roms.vc_h_opinf_operator(Phi, J, train_trajs, train_derivs)
evals = np.linalg.eigvals(A_bar)

print(f"Re(lambda)=[{evals.real.max()}, {evals.real.min()}]")
print(f"Im(lambda)=[{evals.imag.max()}, {evals.imag.min()}]")
print("A_bar positive definite: ", np.all(np.linalg.eigvals(A_bar) > 0))

# ── Reduced dynamics  dz/dt = A z  for each ROM ───────────────────────────────
A_dyn_opinf = wave_roms.opinf_dynamics(Phi, J, A_bar)
A_dyn_PG    = wave_roms.pg_dynamics(Phi, J, W_fom)

# Trained NiTROMs: one labeled container per initialization case,
# output/<BASIS>_<tensor>/ (written by train_nitrom.py); untrained or stale
# containers are skipped with a notice. Each entry holds the reduced dynamics
# Jhat (A2 + A2^T), the symplectic PG encoder/decoder pair, A2 (for the
# conserved reduced energy z^T A2 z), and the container path (history plots).
def load_nitrom(tensor):
    folder   = nitrom_output_dir(case, BASIS, tensor)
    phi_path = os.path.join(folder, "Phi_nit.npy")
    if not os.path.exists(phi_path):
        print(f"[skip] no NiTROM solution in {folder} -> train it with "
              f"train_nitrom.py (BASIS={BASIS!r}, TENSOR={tensor!r})")
        return None
    Phi_nit = np.load(phi_path)
    if Phi_nit.shape[0] != Phi.shape[0]:                 # stale run from a different N
        print(f"[skip] {BASIS}_{tensor}: Phi_nit ({Phi_nit.shape[0]}) != current "
              f"basis ({Phi.shape[0]}); retrain train_nitrom.py.")
        return None
    A2_nit   = np.load(os.path.join(folder, "A2_nit.npy"))
    Jhat_nit = Phi_nit.T @ J @ Phi_nit
    print(f"Loaded NiTROM {BASIS}_{tensor}: Phi {Phi_nit.shape}, A2 {A2_nit.shape}")
    return dict(A_dyn=Jhat_nit @ (A2_nit + A2_nit.T),
                encoder=(J @ Phi_nit).T,
                decoder=-Phi_nit @ np.linalg.inv(Jhat_nit),
                A2=A2_nit, folder=folder)


nitroms = {}
_init_cases = [("opinf", "NiTROM_OpInf")]
if SHOW_NITROM_PG:
    _init_cases.insert(0, ("pg", "NiTROM_PG"))
for _tensor, _key in _init_cases:
    _m = load_nitrom(_tensor)
    if _m is not None:
        nitroms[_key] = _m

# Eigenvalue stability check: dz/dt = A z for each model (FOM included as reference).
# max Re(lambda) > 0 => growth (unstable); max |Im(lambda)| ~ fastest oscillation.
_dynamics = {
    "FOM":    J @ W_fom,        # full-order  x_dot = J W x
    "OpInf":  A_dyn_opinf,
    "PG":     A_dyn_PG,
    **{k: m["A_dyn"] for k, m in nitroms.items()},
}
print(" ---- Eigenvalue stability check (dz/dt = A z) ---- ")
print(f"  {'model':<8} {'max Re(lambda)':>16} {'max |Im(lambda)|':>18}")
for _name, _A in _dynamics.items():
    _ev = np.linalg.eigvals(_A)
    print(f"  {_name:<8} {_ev.real.max():>+16.3e} {np.abs(_ev.imag).max():>18.3e}")
print("-"*50)


# ── Held-out test set: same sampling space as training, different seed ────────
fom         = case.fom
from wave_cases import TEST_SEED, N_TEST             # shared with main.py (plots the same draw)
test_trajs, _, test_params = wave_sampling.generate_trajectories(
    fom, time, N_TEST, seed=TEST_SEED, ranges=case.sample_ranges)
test_trajs = list(test_trajs)                        # match train_trajs (list of (2N, n_t))
print(f"Generated {N_TEST} test trajectories within training sampling space)")

# ── Comparison setup (shared by both datasets) ────────────────────────────────
j_probe     = N // 2                                 # middle node (~ x = L/2)
decoder_op, encoder_op   = Phi, Phi.T
decoder_pg, encoder_pg   = -Phi @ Jhat_inv, (J @ Phi).T
# (NiTROM encoders/decoders live inside the `nitroms` dict entries)


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
# Size the horizon to at least one full wave period (L/c) for the active case,
# so the videos cover a complete cycle instead of cutting off mid-traversal
# (periodic period L/c = 10 was being truncated by the old hard-coded 8.0).
period   = case.L / case.c
T_SIM    = max(8.0, period)
dt       = time[1] - time[0]
time_sim = np.arange(0.0, T_SIM + 0.5 * dt, dt)
A_fom    = J @ W_fom
ns       = len(time_sim)

# Family styling: every model gets a DISTINCT color so overlapping curves stay
# separable (a baseline and its trained NiTROM no longer share a hue). Warm hues
# = PG family (orange baseline, red trained), cool hues = OpInf family (blue
# baseline, green trained); FOM = neutral near-black. Style still reinforces it:
# dashed/light = baseline, solid/heavy = trained.
LABELS = {"FOM": "FOM", "PG": "Petrov-Galerkin", "OpInf": "VC-H-OpInf",
          "NiTROM_PG": "NiTROM (PG-initialized)",
          "NiTROM_OpInf": "NiTROM (OpInf-initialized)"}
COLORS = {"FOM": "#0b0b0b", "PG": "#e8710a", "OpInf": "#2ca02c",
          "NiTROM_PG": "#d62728", "NiTROM_OpInf": "#1f77b4"}
STYLES = {"FOM": "-",  "PG": "--", "OpInf": "--",
          "NiTROM_PG": "-", "NiTROM_OpInf": "-"}
WIDTHS = {"FOM": 2.0,  "PG": 1.4,  "OpInf": 1.4,
          "NiTROM_PG": 2.2, "NiTROM_OpInf": 2.2}

# Video framing (shared). Dirichlet: state holds interior nodes only; pad the
# walls (u=0 at x=0 and x=L) so the BC is visible. Periodic already spans [0, L].
# SLOWED playback: the GIF wall-clock is SLOWDOWN x the physical time, i.e. 1 s
# of video == 1/SLOWDOWN time units of the simulation (SLOWDOWN = 1 -> real
# time). Frames are decimated evenly (capped at a viewer-safe display rate) so
# the clock in the title advances uniformly.
TARGET_FPS = 25                                    # display rate ceiling (viewer-safe)
SLOWDOWN   = 4.0                                   # playback lasts SLOWDOWN * T_SIM seconds
n_frames   = min(ns, max(2, int(round(SLOWDOWN * T_SIM * TARGET_FPS))))
frame_ids  = np.linspace(0, ns - 1, n_frames).astype(int)
fps        = n_frames / (SLOWDOWN * T_SIM)
if TESTCASE == "dirichlet":
    x_plot = np.concatenate(([0.0], x_grid, [x_grid[-1] + x_grid[0]]))
    def pad_bc(q):                          # (N, nt) -> (N+2, nt): zero rows at both walls
        z = np.zeros((1, q.shape[1]))
        return np.vstack([z, q, z])
else:
    x_plot = x_grid
    def pad_bc(q):
        return q


def make_traj_video(trajs, ti, label, tag, out):
    """One q(x, t) animation: FOM vs ROMs for trajectory ``ti`` of ``trajs``."""
    x0 = trajs[ti][:, 0]
    # (key, q(x,t)); FOM first so its amplitude sets the view. Color/style/width per model.
    series = [("FOM",   integrate_linear(A_fom, x0, time_sim)[:N, :]),
              ("PG",    (decoder_pg @ integrate_linear(A_dyn_PG,    encoder_pg @ x0, time_sim))[:N, :]),
              ("OpInf", (decoder_op @ integrate_linear(A_dyn_opinf, encoder_op @ x0, time_sim))[:N, :])]
    for k, m in nitroms.items():
        series.append((k, (m["decoder"] @ integrate_linear(
            m["A_dyn"], m["encoder"] @ x0, time_sim))[:N, :]))

    series = [(k, pad_bc(q)) for k, q in series]  # show BC walls

    fig_v, ax = plt.subplots(figsize=(6, 4.5))
    lines = [(ax.plot(x_plot, q[:, frame_ids[0]], color=COLORS[k], ls=STYLES[k],
                      lw=1.2, label=LABELS[k])[0], q)
             for k, q in series]
    amp = 1.3 * np.max(np.abs(series[0][1])) + 1e-12   # FOM amplitude sets y-range
    ax.set_ylim(-amp, amp); ax.set_xlim(x_plot[0], x_plot[-1])
    ax.set_xlabel("x"); ax.set_ylabel(r"$q(x,\,t)$")
    ax.set_title(f"{label} trajectory {ti} ({BASIS_TAG})")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    suptitle = fig_v.suptitle(f"t = {time_sim[frame_ids[0]]:.3f}")

    def _update(f):
        for ln, q in lines:
            ln.set_ydata(q[:, f])
        suptitle.set_text(f"t = {time_sim[f]:.3f}")
        return []

    anim = FuncAnimation(fig_v, _update, frames=frame_ids,
                         interval=1000.0 / fps, blit=False)
    gif_path = os.path.join(out, f"rom_vs_fom_{tag}_{ti:03d}.gif")
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig_v)
    return gif_path


def evaluate_dataset(trajs, label, tag, out):
    """Compare FOM vs all ROMs on ``trajs`` (a list of (2N, n_t) full-state
    trajectories). ``label`` -> plot titles, ``tag`` -> filename suffix,
    ``out`` -> output subfolder (training/ or testing/).
    Writes the static q(L/2, t) comparison, the per-trajectory and mean
    reconstruction-error curves, and one animation per trajectory."""
    print(f"\n=== Evaluating on {label.lower()} dataset ({len(trajs)} trajectories) ===")
    ntraj       = len(trajs)
    compare_idx = [0, ntraj // 2]
    # Per-trajectory energy weight (norm(H)^0.5), same formula as the training weights.
    alpha = np.array([np.linalg.norm(fom.hamiltonian(trajs[j])) ** 0.5 for j in range(ntraj)])

    # Full-space trajectories stored as 2N x (ntraj*ns); traj j -> cols [j*ns:(j+1)*ns].
    X_fom = np.zeros((2 * N, ntraj * ns))
    X_op  = np.zeros_like(X_fom)
    X_pg  = np.zeros_like(X_fom)
    X_nit = {k: np.zeros_like(X_fom) for k in nitroms}   # per trained container

    H_fom = np.zeros((ntraj, ns))
    H_op  = np.zeros_like(H_fom)
    H_pg  = np.zeros_like(H_fom)
    H_nit = {k: np.zeros_like(H_fom) for k in nitroms}   # z^T A2 z (own invariant)

    for j in range(ntraj):
        x0 = trajs[j][:, 0]
        sl = slice(j * ns, (j + 1) * ns)
        Z_op  = integrate_linear(A_dyn_opinf, encoder_op  @ x0, time_sim)   # (r,  ns)
        X_fom[:, sl] = integrate_linear(A_fom, x0, time_sim)
        X_op[:,  sl] = decoder_op  @ Z_op
        X_pg[:,  sl] = decoder_pg  @ integrate_linear(A_dyn_PG, encoder_pg @ x0, time_sim)
        for k, m in nitroms.items():
            Z_k = integrate_linear(m["A_dyn"], m["encoder"] @ x0, time_sim)  # (r', ns)
            X_nit[k][:, sl] = m["decoder"] @ Z_k
            H_nit[k][j, :]  = np.einsum("ki,ki->i", Z_k, m["A2"] @ Z_k)      # z^T A2 z

        H_fom[j, :] = fom.hamiltonian(X_fom[:, sl])
        H_pg[j, :]  = fom.hamiltonian(X_pg[:, sl])
        H_op[j, :]  = 0.5 * np.einsum("ki,ki->i", Z_op,  A_bar  @ Z_op)    # 0.5 z^T A_bar z

    # --- Energy drift, averaged over all trajectories: mean_j |H_j(t) - H_j(0)| ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for k, Hk in [("FOM", H_fom), ("PG", H_pg), ("OpInf", H_op),
                  *H_nit.items()]:
        drift = np.mean(np.abs(Hk - Hk[:, [0]]), axis=0)              # (ns,) mean over trajectories
        ax.semilogy(time_sim, drift, color=COLORS[k], ls=STYLES[k], lw=WIDTHS[k], label=LABELS[k])
    ax.set_xlabel("time [s]")
    # ax.set_ylabel(r"$\frac{1}{N_{\mathrm{traj}}}\sum_{j}\,|H_j(t) - H_j(0)|$")\
    ax.set_ylabel(r"energy drift")
    ax.set_title(f"Traj-avg Energy Drift vs Time ({label.lower()})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"energy_drift_{tag}.png"), dpi=150)
    fig.savefig(os.path.join(out, f"energy_drift_{tag}.pdf"), bbox_inches="tight")
    plt.close(fig)


    # Error matrix e_j(t_i) = (1/alpha_j) ||X_fom_j - Xhat_j||^2  -> (ntraj, ns).
    def error_matrix(Xhat):
        d = (X_fom - Xhat).reshape(2 * N, ntraj, ns)
        return np.sum(d ** 2, axis=0) / alpha[:, None]
    E_op, E_pg = error_matrix(X_op), error_matrix(X_pg)
    E_nit = {k: error_matrix(Xk) for k, Xk in X_nit.items()}

    # ── Static plot: q(L/2, t), FOM vs ROMs, on the compare_idx trajectories ──
    fig, axes = plt.subplots(1, len(compare_idx), figsize=(11, 4.2), sharex=True)
    for ax, ti in zip(np.atleast_1d(axes), compare_idx):
        sl = slice(ti * ns, (ti + 1) * ns)
        ax.plot(time_sim, X_fom[j_probe, sl], color=COLORS["FOM"],   ls=STYLES["FOM"],   lw=WIDTHS["FOM"],   label=LABELS["FOM"])
        ax.plot(time_sim, X_pg[j_probe, sl],  color=COLORS["PG"],    ls=STYLES["PG"],    lw=WIDTHS["PG"],    label=LABELS["PG"])
        ax.plot(time_sim, X_op[j_probe, sl],  color=COLORS["OpInf"], ls=STYLES["OpInf"], lw=WIDTHS["OpInf"], label=LABELS["OpInf"])
        for k, Xk in X_nit.items():
            ax.plot(time_sim, Xk[j_probe, sl], color=COLORS[k], ls=STYLES[k], lw=WIDTHS[k], label=LABELS[k])
        ax.set_title(f"{label} trajectory {ti} ({BASIS_TAG})")
        ax.set_xlabel("time"); ax.set_ylabel(r"$q(L/2,\,t)$")
        ax.axhline(0, color="0.7", linewidth=0.5)
        fom_amp = np.max(np.abs(X_fom[j_probe, sl]))           # clip y to FOM range
        ax.set_ylim(-1.4 * fom_amp, 1.4 * fom_amp)
        if ti == compare_idx[0]:
            ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    out_png = os.path.join(out, f"compare_fom_opinf_pg_{tag}.png")
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")

    # ── Error vs time (per trajectory) ───────────────────────────────────────
    fig_e, ax_e = plt.subplots(figsize=(5, 4))
    for traj in range(ntraj):
        lab = (traj == 0)   # label once for the legend
        ax_e.semilogy(time_sim, E_pg[traj, :],  lw=0.9, color=COLORS["PG"],     ls=STYLES["PG"],     label=LABELS["PG"]    if lab else None)
        ax_e.semilogy(time_sim, E_op[traj, :],  lw=0.9, color=COLORS["OpInf"],  ls=STYLES["OpInf"],  label=LABELS["OpInf"] if lab else None)
        for k, Ek in E_nit.items():
            ax_e.semilogy(time_sim, Ek[traj, :], lw=0.9, color=COLORS[k], ls=STYLES[k], label=LABELS[k] if lab else None)
        if traj == 0:
            ax_e.legend(loc="upper right", fontsize=11, framealpha=0.90)
    ax_e.set_xlabel("time")
    # ax_e.set_ylabel(r"$\frac{1}{\alpha_j} \ \|x_j(t)-\hat{x}_j(t)\|^2$")
    ax_e.set_ylabel(r"relative error")
    ax_e.set_title(f"State error vs time, {label.lower()}, {BASIS_TAG})")
    # ax_e.grid(True, which="both", alpha=0.3)
    fig_e.tight_layout()
    err_png = os.path.join(out, f"error_vs_time_{tag}.png")
    fig_e.savefig(err_png, dpi=150)
    fig_e.savefig(err_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig_e)
    print(f"Saved {err_png}")

    # ── Error vs time (trajectory-averaged) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(time_sim, E_pg.mean(axis=0), color=COLORS["PG"],    ls=STYLES["PG"],    label="Petrov-Galerkin")
    ax.semilogy(time_sim, E_op.mean(axis=0), color=COLORS["OpInf"], ls=STYLES["OpInf"], label="VC-H-OpInf")
    for k, Ek in E_nit.items():
        ax.semilogy(time_sim, Ek.mean(axis=0), color=COLORS[k], ls=STYLES[k], label=LABELS[k])
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"relative error")
    ax.set_title(f"Traj-avg Error vs Time ({label.lower()})")
    # ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.90)
    fig.tight_layout()
    err_png = os.path.join(out, f"error_vs_time_mean_{tag}.png")
    fig.savefig(err_png, dpi=150)
    fig.savefig(err_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {err_png}")

    # ── Videos: one GIF per trajectory ───────────────────────────────────────
    # for ti in range(len(trajs)):
    for ti in range(4):
        print(f"Saved {make_traj_video(trajs, ti, label, tag, out)}")


# ── Run the comparison on BOTH datasets ───────────────────────────────────────
evaluate_dataset(train_trajs, "Training", "train", train_out)
evaluate_dataset(test_trajs,  "Testing",  "test",  test_set_out)


# ── NiTROM training histories (one per trained container) ─────────────────────
for k, m in nitroms.items():
    hist = {f: os.path.join(m["folder"], f)
            for f in ("costvec_nit.npy", "gradvec_nit.npy", "itervec_nit.npy")}
    if not all(os.path.exists(p) for p in hist.values()):
        print(f"[skip] no cost/grad history in {m['folder']}")
        continue
    cost_hist = np.load(hist["costvec_nit.npy"])
    grad_hist = np.load(hist["gradvec_nit.npy"])
    iters     = np.load(hist["itervec_nit.npy"])
    if len(iters) != len(cost_hist):                # fall back to a plain index
        iters = np.arange(len(cost_hist))

    figh, (axc, axg) = plt.subplots(1, 2, figsize=(11, 4.2))
    axc.semilogy(iters, cost_hist, "k-", lw=1.5)
    axc.set_xlabel("iteration"); axc.set_ylabel("cost")
    axc.set_title(f"{LABELS[k]} training cost ({BASIS_TAG})")
    axg.semilogy(iters, grad_hist, "k-", lw=1.5)
    axg.set_xlabel("iteration"); axg.set_ylabel("gradient norm")
    axg.set_title(f"{LABELS[k]} gradient norm ({BASIS_TAG})")
    figh.tight_layout()
    hist_png = os.path.join(test_out, f"training_history_{k}.png")
    figh.savefig(hist_png, dpi=150)
    figh.savefig(hist_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(figh)
    print(f"Saved {hist_png}")
