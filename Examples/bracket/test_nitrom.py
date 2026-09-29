"""
Evaluate the ROMs against the FOM on the bracket, for ONE choice of reduced
basis (BASIS = "cl" | "pod"). Four reduced models are compared:

    PG            Petrov-Galerkin projection of the FOM operator (intrusive)
    VC-H-OpInf    variationally-consistent Hamiltonian OpInf     (learned)
    NiTROM-PG     NiTROM trained from the PG initialization
    NiTROM-OpInf  NiTROM trained from the VC-H-OpInf initialization

The two NiTROM solutions are read from their labeled containers
Bracket/output/<basis>_pg and <basis>_opinf (written by train_nitrom.py);
containers without a trained solution are skipped with a notice.

The FOM trajectory is loaded from disk (the state dim ~1e4 makes a dense
matrix-exponential propagation infeasible); only the small reduced ROMs are
integrated here, symplectically. Outputs -- relative-error-vs-time,
energy-drift, tip-displacement and FOM-vs-ROM natural-frequency plots -- go to
test_results/<basis>/.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import seaborn as sns
sns.set(style="ticks", font_scale=1.2, palette="tab10")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from plate_cases import get_case, canonical_J, nitrom_output_dir
from NiTROM.Optimization_Functions.linear_symplectic_integrators import linear_symplectic_solve
from NiTROM.Optimization_Functions.classes import HamiltonianROM
import plate_roms

# ============================================================================
# Configuration
# ============================================================================
TESTCASE = "bracket"          # which testcase to run: "bracket" | "cantilever_plate"
BASIS    = "cl"               # reduced basis used by ALL models: "cl" | "pod"
SHOW_NITROM_PG = False        # plot the PG-initialized NiTROM too (OpInf is the
                              # better init here, so only the OpInf-initialized
                              # NiTROM is shown). The PG/OpInf baselines always show.

# Internal symplectic step (match train_nitrom's dt_rom). Chosen COMMENSURATE
# with the snapshot spacing (1e-4 = 4000 * DT_ROM) so every snapshot time is an
# internal integration knot: the spline returns the exact symplectic state at
# the data times (the spline itself is not symplectic between knots).
DT_ROM = 2.5e-8

BASIS_LABEL = {"cl": "CL", "pod": "POD"}[BASIS]

case      = get_case(TESTCASE)
train_dir = case.data_dir
test_out  = os.path.join(case.test_results_dir, BASIS)   # plots for THIS basis
os.makedirs(test_out, exist_ok=True)

# ============================================================================
# Load basis, FOM operator, trajectory data
# ============================================================================
Phi   = np.load(os.path.join(train_dir, f"Phi_{BASIS}.npy"))
N2    = Phi.shape[0]
J     = canonical_J(N2)                                    # sparse 2N x 2N
A_fom = load_npz(os.path.join(train_dir, "A_fom.npz"))     # blkdiag(K, M^-1), H = 1/2 x^T A x
print(f"Basis : {BASIS_LABEL}  Phi {Phi.shape};  A_fom {A_fom.shape}")

time    = np.load(os.path.join(train_dir, "time.npy"))              # FULL time grid
n_train = int(np.load(os.path.join(train_dir, "n_train.npy"))[0])   # train/test snapshot split
t_split = time[n_train]                                             # boundary time (vertical line)

X_full   = np.load(os.path.join(train_dir, "x.npy"))                # FULL trajectory (evaluation)
X_train  = X_full[:, :n_train]                                      # TRAIN snapshots (build operators)
Xt_train = np.load(os.path.join(train_dir, "xdot.npy"))[:, :n_train]
print(f"Loaded trajectory {X_full.shape}; train {n_train}/{X_full.shape[1]} snapshots "
      f"(t_split = {t_split:.4g})")

# Full-space recovery for tip-displacement plots: Uc lifts a compressed field to
# the 3N physical DOFs; tip_node picks the node to probe. (Uc absent => no compression.)
tip = int(np.load(os.path.join(train_dir, "tip_node.npy"))[0])
_uc_path = os.path.join(train_dir, "Uc.npy")
Uc = np.load(_uc_path) if os.path.exists(_uc_path) else None
# Uc is a compression encoder (3N, rc) that lifts a COMPRESSED field (rc modes)
# to the 3N physical DOFs. It is valid only when the loaded state is compressed,
# i.e. its field dim (N2//2) equals Uc's input dim (rc = Uc.shape[1]). On full/
# uncompressed data the field dim is 3N == Uc.shape[0], so a leftover Uc.npy from
# an old compressed run is stale -- drop it here so lift_q stays the identity.
if Uc is not None and Uc.shape[1] != N2 // 2:
    print(f"[info] ignoring stale Uc.npy {Uc.shape}: data is uncompressed "
          f"(state field dim {N2 // 2} != Uc input dim {Uc.shape[1]}).")
    Uc = None

# Full-order (uncompressed) FOM energy H(t) over the full trajectory (raw
# Exodus DATA, no integration), on the data time grid.
fom_energy_full = np.load(os.path.join(train_dir, "fom_energy.npy"))

# ============================================================================
# Reduced models: baselines (PG, VC-H-OpInf) + trained NiTROMs
# ============================================================================
# Every model is a dict with the same four entries, so the evaluation below
# treats them uniformly:
#   A_dyn    reduced dynamics  dz/dt = A_dyn z
#   encoder  full state  -> reduced state   (z = encoder @ x)
#   decoder  reduced state -> full state    (x ~ decoder @ z)
#   W        reduced energy operator: the model conserves H(z) = 1/2 z^T W z


def make_definite(A, sign="pos", floor=1e-6):
    """Nearest definite matrix to symmetric A: same eigenvectors, clipped eigenvalues."""
    w, U = np.linalg.eigh(0.5 * (A + A.T))       # real eigenvalues (ascending), orthonormal U
    thr = floor * np.abs(w).max()                # relative floor, not absolute
    if sign == "pos":
        w = np.clip(w, thr, None)                # all >= +thr  -> SPD
    else:                                        # sign == "neg"
        w = np.clip(w, None, -thr)               # all <= -thr  -> negative definite
    return U @ np.diag(w) @ U.T


# Reduced symplectic structure of the chosen basis (canonical for CL, cond = 1;
# generic-but-invertible for POD).
Jhat       = Phi.T @ np.asarray(J @ Phi)
_cond_Jhat = np.linalg.cond(Jhat)
print(f"cond(Jhat) = {_cond_Jhat:.3e}")
if Phi.shape[1] % 2 or _cond_Jhat > 1e12:
    raise ValueError(
        f"Jhat = Phi^T J Phi is (numerically) singular: r = {Phi.shape[1]}"
        f"{' (ODD -> singular by skew-symmetry)' if Phi.shape[1] % 2 else ''}, "
        f"cond(Jhat) = {_cond_Jhat:.3e}. A symplectic ROM needs an even r with "
        f"invertible Jhat -- regenerate the basis (prepare_data.py caps and "
        f"evens the POD rank).")
Jhat_inv = np.linalg.inv(Jhat)
# Petrov-Galerkin: intrusive projection of A_fom; symplectic encoder/decoder pair.
decoder_pg = -Phi @ Jhat_inv
models = {
    "PG": dict(
        A_dyn=plate_roms.pg_dynamics(Phi, J, A_fom),
        encoder=np.asarray(J @ Phi).T,
        decoder=decoder_pg,
        W=decoder_pg.T @ np.asarray(A_fom @ decoder_pg),   # projected physical energy
    ),
}

# VC-H-OpInf: learned SPD operator A_bar (range-restricted Lyapunov solve on the
# TRAINING snapshots), Galerkin encoder/decoder.
A_bar = plate_roms.vc_h_operator(Phi, J, [X_train], [Xt_train])
A_bar = make_definite(A_bar, sign="pos")                   # SPD -> imaginary spectrum
models["OpInf"] = dict(
    A_dyn=plate_roms.opinf_dynamics(Phi, J, A_bar),
    encoder=Phi.T,
    decoder=Phi,
    W=A_bar,                                               # OpInf's learned invariant
)


def load_nitrom(tensor):
    """Trained NiTROM from the container output/<BASIS>_<tensor>, or None.

    The trained pair (Phi_nit, A2_nit) gives the symplectic Petrov-Galerkin
    encoder/decoder and the dynamics A_dyn = Jhat (A2 + A2^T); the conserved
    reduced energy is H(z) = 1/2 z^T (A2 + A2^T) z.
    """
    folder = nitrom_output_dir(case, BASIS, tensor)
    phi_path = os.path.join(folder, "Phi_nit.npy")
    if not os.path.exists(phi_path):
        print(f"[skip] no NiTROM solution in {folder} -> train it with "
              f"train_nitrom.py (BASIS={BASIS!r}, TENSOR={tensor!r})")
        return None
    Phi_nit  = np.load(phi_path)
    if Phi_nit.shape[0] != N2:
        print(f"[skip] {BASIS}_{tensor}: solution in {folder} has state dim "
              f"{Phi_nit.shape[0]} but the current data is {N2}-dimensional -- "
              f"it was trained under a different COMPRESS setting (prepare_data.py). "
              f"Retrain this case on the current data.")
        return None
    A2_nit   = np.load(os.path.join(folder, "A2_nit.npy"))
    Jhat_nit = Phi_nit.T @ np.asarray(J @ Phi_nit)
    print(f"Loaded NiTROM {BASIS}_{tensor}: Phi {Phi_nit.shape}, A2 {A2_nit.shape}")
    return dict(
        A_dyn=Jhat_nit @ (A2_nit + A2_nit.T),
        encoder=np.asarray(J @ Phi_nit).T,
        decoder=-Phi_nit @ np.linalg.inv(Jhat_nit),
        W=A2_nit + A2_nit.T,
        # source-code Hamiltonian, used to cross-check the closed form above
        rom=HamiltonianROM(operators=[A2_nit], poly_comp=[1], Phi=Phi_nit),
    )


_init_cases = [("opinf", "NiTROM_OpInf")]
if SHOW_NITROM_PG:
    _init_cases.insert(0, ("pg", "NiTROM_PG"))
for tensor, key in _init_cases:
    nitrom = load_nitrom(tensor)
    if nitrom is not None:
        models[key] = nitrom

# ── Spectra: conservative models must have purely imaginary eigenvalues ───────
fom_freq_path = os.path.join(train_dir, "fom_freqs.npy")
if os.path.exists(fom_freq_path):
    w_min, w_max = np.load(fom_freq_path)      # precomputed in prepare_data (sparse eigsh)
    print(f"FOM dynamics:  |Im| band = [{w_min:.3e}, {w_max:.3e}]  (Re = 0)")
for k, m in models.items():
    ev = np.linalg.eigvals(m["A_dyn"])
    print(f"{k:13s} dynamics: max Re = {ev.real.max():+.3e}, max |Im| = {np.abs(ev.imag).max():.3e}")

# Spectrum of the VC-H-OpInf dynamics in the complex plane. A conservative
# Hamiltonian ROM has eigenvalues on the imaginary axis (Re = 0); nonzero Re
# means growth/decay. The FOM band |Im| in [w_min, w_max] is shown for reference.
opinf_ev = np.linalg.eigvals(models["OpInf"]["A_dyn"])
fig_ev, ax_ev = plt.subplots(figsize=(6, 4.5))
ax_ev.axvline(0, color="0.7", lw=0.8, zorder=0)            # imaginary axis (Re = 0)
if os.path.exists(fom_freq_path):
    for w in (w_min, w_max):                               # FOM band edges (+/-)
        ax_ev.axhline(+w, color="0.6", ls=":", lw=1.0, zorder=0)
        ax_ev.axhline(-w, color="0.6", ls=":", lw=1.0, zorder=0)
    ax_ev.plot([], [], color="0.6", ls=":", label=r"FOM $|\mathrm{Im}|$ band")
ax_ev.scatter(opinf_ev.real, opinf_ev.imag, s=40, edgecolor="k", lw=0.5,
              label="VC-H-OpInf", zorder=3)
ax_ev.set_xlabel(r"$\mathrm{Re}(\lambda)$")
ax_ev.set_ylabel(r"$\mathrm{Im}(\lambda)$")
ax_ev.set_title(f"Spectrum of $A_{{\\mathrm{{dyn}}}}$, VC-H-OpInf ({BASIS_LABEL} basis)")
ax_ev.ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
ax_ev.legend(loc="upper right", framealpha=0.95)
fig_ev.tight_layout()
for ext in ("png", "pdf"):
    fig_ev.savefig(os.path.join(test_out, f"opinf_spectrum.{ext}"),
                   dpi=150, bbox_inches="tight")
plt.close(fig_ev)

# ============================================================================
# Styling (one entry per curve; NiTROMs keyed by their initialization)
# ============================================================================
# Every model gets a DISTINCT color so overlapping curves stay separable (a
# baseline and its trained NiTROM no longer share a hue). Warm hues = PG family
# (orange baseline, red trained), cool hues = OpInf family (blue baseline, green
# trained); FOM = neutral near-black. Style reinforces it: dashed/light =
# baseline, solid/heavy = trained. (Matches the wave-equation test_nitrom.)
LABELS = {"FOM": "FOM",       "PG": "PG",      "OpInf": "VC-H-OpInf",
          "NiTROM_PG": "NiTROM (PG-initialized)",
          "NiTROM_OpInf": "NiTROM (OpInf-initialized)"}

COLORS = {"FOM": "#0b0b0b", "PG": "#e8710a", "OpInf": "#2ca02c",
          "NiTROM_PG": "#d62728", "NiTROM_OpInf": "#1f77b4"}

STYLES = {"FOM": "-",                                  # solid
          "PG": "--",          "OpInf": "--",          # baselines: dashed
          "NiTROM_PG": "-",    "NiTROM_OpInf": "-"}    # trained:   solid

WIDTHS = {"FOM": 2.0,
          "PG": 1.1,           "OpInf": 1.1,           # baselines: lighter
          "NiTROM_PG": 2.2,    "NiTROM_OpInf": 2.2}    # trained:   heavier

# COLORS = {"FOM": "#000000",   "PG": "#eca131ff", "OpInf": "#3088c8",
#           "NiTROM_PG": "#d62728",              "NiTROM_OpInf": "#d62728"}
# STYLES = {"FOM": "-",         "PG": "-",       "OpInf": "-",
#           "NiTROM_PG": "-",      "NiTROM_OpInf": "-"}
# WIDTHS = {"FOM": 1.6,         "PG": 1.2,       "OpInf": 1.2,
#           "NiTROM_PG": 1.6,                    "NiTROM_OpInf": 1.6}


# ============================================================================
# Frequency comparison: FOM eigenfrequencies vs each ROM's reduced frequencies
# ============================================================================
# A conservative (Hamiltonian) system has purely imaginary eigenvalues +/- i*w;
# the natural frequencies are w = |Im(lambda)|. Each reduced model (r states) has
# r/2 frequencies; the FOM reference is its LOWEST r/2 eigenfrequencies of the
# Hamiltonian matrix S = J A (the modes an energy-based basis is meant to
# capture). Both are sorted ascending and compared index-by-index. NiTROM-PG is
# included automatically iff SHOW_NITROM_PG is on (it is simply absent from
# `models` otherwise).
def rom_freqs(A_dyn):
    """Sorted natural frequencies w = |Im(lambda)| (rad/s), one per +/-iw pair."""
    w = np.sort(np.abs(np.linalg.eigvals(A_dyn).imag))
    return w[::2]                                    # dedupe the conjugate pairs

# Each model carries its OWN reduced rank (the baselines use the current basis;
# a trained NiTROM may have a different rank if the basis changed since training),
# so every model is plotted against its own mode count. The FOM reference is its
# lowest r_ref = widest-ROM eigenfrequencies of S = J A.
rom_w  = {k: rom_freqs(m["A_dyn"]) for k, m in models.items()}
r_ref  = max((len(w) for w in rom_w.values()), default=Phi.shape[1] // 2)

# FOM reference: lowest r_ref eigenfrequencies of S = J A. Cached to disk (the
# sparse shift-invert solve is the only nontrivial cost); derived purely from
# A_fom -- delete fom_freqs_modes.npy to force a recompute after new data.
_fom_modes_path = os.path.join(train_dir, "fom_freqs_modes.npy")
fom_w = np.load(_fom_modes_path) if os.path.exists(_fom_modes_path) else None
if fom_w is not None and len(fom_w) < r_ref:
    fom_w = None                                     # stale/too-short -> recompute
if fom_w is None:
    try:
        from scipy.sparse.linalg import eigs as _sp_eigs
        S = (J @ A_fom).tocsc()
        ev = _sp_eigs(S, k=2 * r_ref, sigma=0.0, which="LM",   # near 0 -> lowest w
                      return_eigenvectors=False)
        fom_w = np.sort(np.abs(ev.imag))[::2]
        np.save(_fom_modes_path, fom_w)
        print(f"FOM reference: computed lowest {len(fom_w)} eigenfrequencies of S=J*A")
    except Exception as e:
        print(f"[warn] could not compute FOM eigenfrequencies ({e}); "
              f"showing the [w_min, w_max] band only.")

fig_f, ax_f = plt.subplots(figsize=(7, 4.5))
if fom_w is not None:
    ax_f.plot(np.arange(1, len(fom_w) + 1), fom_w, marker="o", ms=5, ls=STYLES["FOM"],
              lw=WIDTHS["FOM"], color=COLORS["FOM"], label=LABELS["FOM"], zorder=5)
elif os.path.exists(fom_freq_path):
    ax_f.axhspan(w_min, w_max, color="0.85", zorder=0, label=r"FOM $|\mathrm{Im}|$ band")
for k, w in rom_w.items():
    ax_f.plot(np.arange(1, len(w) + 1), w, marker="o", ms=4, ls=STYLES[k],
              lw=WIDTHS[k], color=COLORS[k], label=LABELS[k])
ax_f.set_yscale("log")
ax_f.set_xlabel("mode index (sorted ascending)")
ax_f.set_ylabel(r"natural frequency  $\omega = |\mathrm{Im}(\lambda)|$  [rad/s]")
ax_f.set_title(f"FOM vs ROM natural frequencies ({BASIS_LABEL} basis)")
ax_f.legend(fontsize=9, framealpha=0.95)
fig_f.tight_layout()
for ext in ("png", "pdf"):
    fig_f.savefig(os.path.join(test_out, f"frequencies_{BASIS}.{ext}"),
                  dpi=150, bbox_inches="tight")
plt.close(fig_f)


# ============================================================================
# Helpers
# ============================================================================
def lift_q(Xc):
    """Full-space displacement q (3N, nt) from a (possibly compressed) state [q; p].
    q_full = Uc @ q_compressed when compressed; identity otherwise."""
    q = Xc[: Xc.shape[0] // 2]
    return Uc @ q if Uc is not None else q


def tip_disp(Xc):
    """x,y,z displacement at the tip node over time -> (3, nt)."""
    return lift_q(Xc)[3 * tip: 3 * tip + 3, :]


def mark_split(ax, t_split):
    """Train/test divider as an ANNOTATION (not a legend entry): a faint vertical
    line with 'train'/'test' text just inside the top of the axes box."""
    ax.axvline(t_split, color="0.5", ls="--", lw=1.0, zorder=0)
    tr = ax.get_xaxis_transform()
    ax.text(t_split, 0.98, "train ", transform=tr, ha="right", va="top", fontsize=12, color="0.4")
    ax.text(t_split, 0.98, " test",  transform=tr, ha="left",  va="top", fontsize=12, color="0.4")


def fmt_time_axis(ax):
    """Compact time axis: scientific offset so the closely-spaced tick labels
    (t ~ 1e-2 with 4 decimals) don't overlap."""
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(9)


# ============================================================================
# Evaluation: integrate every reduced model, compare to the loaded FOM data
# ============================================================================
def evaluate_dataset(X_fom, t, t_split, tag="full"):
    """Integrate the reduced ROMs from x0, compare to the (loaded) FOM trajectory."""
    x0 = X_fom[:, 0]

    # Integrate every model ONCE, keeping BOTH outputs of the solver:
    #   Z        spline-evaluated on the data grid t  -> plots / errors vs data ONLY
    #   Z_symp   raw symplectic states on the internal dt grid -> ALL energy audits
    # (the cubic spline is not symplectic, so invariants are never computed from it).
    # The internal grid (~1e6 columns) is thinned to <= ~5000 columns right away:
    # every kept column is still an exact symplectic state, and the full fine
    # array is freed before the next model integrates.
    def integ(A, z0):
        Z, t_fine, Z_fine = linear_symplectic_solve(A, np.asarray(z0, dtype=float), t,
                                                    dt=DT_ROM, return_internal=True)
        stride = max(1, len(t_fine) // 5000)
        return Z, t_fine[::stride], Z_fine[:, ::stride].copy()

    # The FOM is NEVER re-integrated: its energy reference is the stored raw
    # Exodus data energy (fom_energy_full), and t_H (the reduced-model energy time
    # axis) is the thinned internal grid, taken from any reduced model below --
    # they all share the same (t, DT_ROM) grid. Re-integrating the full 15828-dim
    # FOM here would allocate a ~100 GB internal array and take hours.
    recon, H_red = {}, {}
    t_H = None
    for k, m in models.items():
        Z_k, t_k, Z_symp = integ(m["A_dyn"], m["encoder"] @ x0)
        if t_H is None:
            t_H = t_k
        recon[k] = m["decoder"] @ Z_k                        # full-state reconstruction
        H_red[k] = 0.5 * np.einsum("it,it->t", Z_symp, m["W"] @ Z_symp)
        if "rom" in m:   # NiTROM: cross-check source compute_hamiltonian vs closed form
            H_src = np.array([m["rom"].compute_hamiltonian(Z_symp[:, j])
                              for j in range(Z_symp.shape[1])])
            print(f"[{k}] H source vs closed-form  max|diff| = "
                  f"{np.abs(H_src - H_red[k]).max():.3e}")

    rel = lambda H: (np.abs(H - H[0]) / abs(H[0])).max()
    print(f"FOM rel. energy drift (Exodus data):  {rel(fom_energy_full):.2e}")

    # ── Relative error vs time ────────────────────────────────────────────────
    # Per-snapshot relative error  e(t) = ||x(t) - x̂(t)||_2 / ||x(t)||_2 :
    # each snapshot is normalized by its own norm, so every time instant counts
    # equally (unlike the Frobenius-ratio metric printed below, which weights
    # snapshots by their energy).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, Xk in recon.items():
        e2 = np.sum((X_fom - Xk) ** 2, axis=0) / np.sum(X_fom ** 2, axis=0)
        ax.semilogy(t[1:], np.sqrt(e2[1:]), color=COLORS[k], ls=STYLES[k], lw=WIDTHS[k],
                    label=rf"{LABELS[k]}")
    mark_split(ax, t_split)
    ax.set_xlabel("time [s]")
    # ax.set_ylabel(r"$\frac{\|x(t)-\hat{x}(t)\|_2}{\|x(t)\|_2}$")
    ax.set_ylabel(r"relative error")
    ax.set_title(f"Reconstruction Error vs Time ({BASIS_LABEL} basis)")
    fmt_time_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(test_out, f"error_vs_time_{tag}_{BASIS}.png"), dpi=150)
    fig.savefig(os.path.join(test_out, f"error_vs_time_{tag}_{BASIS}.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Energy drift: each model's own conserved energy + FOM references ─────
    # One curve per model: the drift |H(t) - H(0)| of the quadratic energy each
    # conserves at the reduced level (the W operator above). All model energies
    # come from the *_symp states on t_H -- never from the splined Z's.
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    drift = lambda H: np.abs(H - H[0]) + 1e-30

    # FOM reference: the raw Exodus DATA energy H(t) = 1/2 x^T A x (full,
    # uncompressed state; fom_energy.npy from prepare_data.py). Its drift
    # reflects the FE solver, not any re-integration. zorder=3 keeps the FOM
    # curve ON TOP of the ROM curves (lines default to zorder=2) while staying
    # first in the legend.
    if len(fom_energy_full) == len(t):
        ax.semilogy(t[1:], drift(fom_energy_full)[1:],
                    color=COLORS["FOM"], lw=WIDTHS["FOM"], label="FOM", zorder=3)

    for k, H_k in H_red.items():
        ax.semilogy(t_H[1:], drift(H_k)[1:], color=COLORS[k], ls=STYLES[k],
                    lw=WIDTHS[k], label=LABELS[k])

    mark_split(ax, t_split)
    ax.set_xlabel("time [s]"); ax.set_ylabel(r"energy drift")
    ax.set_title(f"Energy Drift vs Time ({BASIS_LABEL} basis)")
    fmt_time_axis(ax)
    ax.set_ylim(1e-14, 1e-5)
    ax.legend(fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(test_out, f"energy_drift_{tag}_{BASIS}.png"), dpi=150)
    fig.savefig(os.path.join(test_out, f"energy_drift_{tag}_{BASIS}.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Tip displacement in the FULL physical space (node = center of free end) ─
    # ROM states are decoded to the compressed full state, then lifted to 3N DOFs
    # via Uc, and the tip node's dominant displacement component is plotted.
    U_fom = tip_disp(X_fom)                                  # loaded data = ground truth
    comp  = int(np.argmax(np.abs(U_fom).max(axis=1)))        # dominant x/y/z component
    # Sanity check vs the ~0.2 body extent: report peak |tip displacement| per model.
    print(f"[tip] peak |u_{'xyz'[comp]}|  FOM(data)={np.abs(U_fom[comp]).max():.3e}  " +
          "  ".join(f"{LABELS[k]}={np.abs(tip_disp(Xk)[comp]).max():.3e}" for k, Xk in recon.items()))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # FOM drawn with a higher zorder so it sits ON TOP of the ROM curves
    # (lines default to zorder=2) while keeping first place in the legend.
    ax.plot(t, U_fom[comp], color=COLORS["FOM"], lw=WIDTHS["FOM"],
            label="FOM (data)", zorder=3)
    for k, Xk in recon.items():
        ax.plot(t, tip_disp(Xk)[comp], color=COLORS[k], ls=STYLES[k],
                lw=WIDTHS[k], label=LABELS[k])
    mark_split(ax, t_split)
    ax.set_xlabel("t")
    ax.set_ylabel(rf"tip $q_{'xyz'[comp]}$")
    ax.set_title(f"Tip displacement ({BASIS_LABEL} basis)")
    fmt_time_axis(ax)
    ax.legend(fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(test_out, f"tip_disp_{tag}_{BASIS}.png"), dpi=150)
    fig.savefig(os.path.join(test_out, f"tip_disp_{tag}_{BASIS}.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Zoom of previous plot with y axis limit set to +-0.05
    fig, ax = plt.subplots(figsize=(6, 4.5))
    # FOM drawn with a higher zorder so it sits ON TOP of the ROM curves
    # (lines default to zorder=2) while keeping first place in the legend.
    ax.plot(t, U_fom[comp], color=COLORS["FOM"], lw=WIDTHS["FOM"],
            label="FOM (data)", zorder=3)
    for k, Xk in recon.items():
        ax.plot(t, tip_disp(Xk)[comp], color=COLORS[k], ls=STYLES[k],
                lw=WIDTHS[k], label=LABELS[k])
    mark_split(ax, t_split)
    ax.set_xlabel("t")
    ax.set_ylabel(rf"tip $q_{'xyz'[comp]}$")
    ax.set_title(f"Tip displacement ({BASIS_LABEL} basis)")
    fmt_time_axis(ax)
    ax.set_ylim(-0.05, 0.05)
    ax.legend(fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(test_out, f"tip_disp_{tag}_{BASIS}_zoom.png"), dpi=150)
    fig.savefig(os.path.join(test_out, f"tip_disp_{tag}_{BASIS}_zoom.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── FFT spectrum: tip-motion frequency content, FOM data vs each ROM ──────
    # FFT of the tip-displacement signal on the DATA grid (dt = t[1]-t[0]); power
    # |u_hat(f)|^2 is summed over x/y/z so no direction is missed, DC removed, and
    # a Hann window applied to cut spectral leakage. This answers "does the ROM
    # reproduce the FOM's dominant spectral peaks?".
    # CAVEAT: the data is sampled at dt, so this resolves only up to the Nyquist
    # f = 1/(2 dt) (~5 kHz here). The FOM's fast modes sit far above Nyquist and
    # ALIAS -- FOM and ROMs fold identically (same grid), so matching peaks is
    # still meaningful, but only the low-frequency band is physical. The TRUE
    # per-mode frequencies (to MHz) are in the frequencies_<basis> eigenvalue plot.
    dt_fft = t[1] - t[0]
    freqs  = np.fft.rfftfreq(len(t), d=dt_fft)             # Hz
    f_nyq  = 0.5 / dt_fft
    win    = np.hanning(len(t))
    def tip_psd(X):
        u = tip_disp(X)                                    # (3, nt), physical tip motion
        u = u - u.mean(axis=1, keepdims=True)              # drop DC
        return np.sum(np.abs(np.fft.rfft(u * win, axis=1)) ** 2, axis=0)  # (nf,) total power
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(freqs, tip_psd(X_fom) + 1e-30, color=COLORS["FOM"],
                lw=WIDTHS["FOM"], label="FOM (data)", zorder=3)
    for k, Xk in recon.items():
        ax.semilogy(freqs, tip_psd(Xk) + 1e-30, color=COLORS[k], ls=STYLES[k],
                    lw=WIDTHS[k], label=LABELS[k])
    ax.set_xlabel(f"frequency  [Hz]   (Nyquist {f_nyq:.0f} Hz)")
    ax.set_ylabel(r"tip PSD  $\sum_{xyz}|\widehat{u}(f)|^2$")
    ax.set_title(f"Tip-motion spectrum, {tag} ({BASIS_LABEL} basis)")
    ax.set_xlim(0, freqs[-1])
    ax.legend(fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(test_out, f"fft_spectrum_{tag}_{BASIS}.png"), dpi=150)
    fig.savefig(os.path.join(test_out, f"fft_spectrum_{tag}_{BASIS}.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Relative L2 error over the whole trajectory, and split train / test ──
    ns = X_fom.shape[1]
    n_tr = int(np.argmin(np.abs(t - t_split)))
    def relerr(sl):
        return {k: float(np.linalg.norm((X_fom - Xk)[:, sl]) / np.linalg.norm(X_fom[:, sl]))
                for k, Xk in recon.items()}
    for name, sl in (("train", slice(0, n_tr)), ("test", slice(n_tr, ns)), ("all", slice(0, ns))):
        e = relerr(sl)
        print(f"[{name:5s}] relative L2 error:  " +
              "  ".join(f"{LABELS[k]}={v:.3e}" for k, v in e.items()))


# ── Run once over the full trajectory (train/test split shown as a vertical line)
evaluate_dataset(X_full, time, t_split)

print(f"Saved evaluation plots to {test_out}")
