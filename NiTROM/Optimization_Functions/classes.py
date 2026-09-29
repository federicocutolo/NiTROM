import numpy as np 
import os
import time as time_module
import importlib.util

try:
    from mpi4py import MPI
except ModuleNotFoundError:  # pragma: no cover - optional dependency for non-MPI workflows
    MPI = None
from itertools import combinations
from string import ascii_lowercase as ascii
from scipy.sparse import bmat, eye, csr_matrix
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

try:
    from .symplectic_integrators import symplectic_integrate, symplectic_solve
except ImportError:  # pragma: no cover - supports direct module loading in tests
    _symplectic_path = os.path.join(
        os.path.dirname(__file__),
        "symplectic_integrators.py",
    )
    _symplectic_spec = importlib.util.spec_from_file_location(
        "nitrom_symplectic_integrators",
        _symplectic_path,
    )
    _symplectic_module = importlib.util.module_from_spec(_symplectic_spec)
    _symplectic_spec.loader.exec_module(_symplectic_module)
    symplectic_integrate = _symplectic_module.symplectic_integrate
    symplectic_solve = _symplectic_module.symplectic_solve

try:
    from .linear_symplectic_integrators import linear_symplectic_solve, linear_propagator
except ImportError:  # pragma: no cover - supports direct module loading in tests
    _lin_path = os.path.join(os.path.dirname(__file__), "linear_symplectic_integrators.py")
    _lin_spec = importlib.util.spec_from_file_location("nitrom_linear_symplectic_integrators", _lin_path)
    _lin_module = importlib.util.module_from_spec(_lin_spec)
    _lin_spec.loader.exec_module(_lin_module)
    linear_symplectic_solve = _lin_module.linear_symplectic_solve
    linear_propagator = _lin_module.linear_propagator


_RUNTIME_PROFILE = {}


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def timing_enabled():
    return _env_flag("NITROM_TIMING")


def debug_enabled():
    return _env_flag("NITROM_DEBUG")


def reset_runtime_profile():
    if timing_enabled():
        _RUNTIME_PROFILE.clear()


def record_runtime(key, dt):
    if not timing_enabled():
        return
    stats = _RUNTIME_PROFILE.setdefault(key, {"time": 0.0, "count": 0})
    stats["time"] += dt
    stats["count"] += 1


def format_runtime_profile():
    if not _RUNTIME_PROFILE:
        return "no timing data collected"

    parts = []
    for key in sorted(_RUNTIME_PROFILE):
        stats = _RUNTIME_PROFILE[key]
        parts.append(f"{key}: {stats['time']:.3e}s ({stats['count']} calls)")
    return ", ".join(parts)

class mpi_pool:

    def __init__(self,comm,n_traj,fname_traj,fname_time,**kwargs):
        """
        Initialize MPI pool and load distributed training data.

        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator.
        n_traj : int
            Total number of trajectories to load from disk.
        fname_traj : str
            Filename pattern used to load each trajectory, e.g. 'traj_%03d.npy'.
        fname_time : str
            Filename for the time vector, e.g. 'time.txt'.
        **kwargs : optional
            fname_weights : str, optional
                Filename pattern for weights, e.g. 'weight_%03d.npy'.
            fname_steady_forcing : str, optional
                Filename pattern for steady forcing, e.g. 'forcing_%03d.npy'.
            fname_derivs : str, optional
                Filename pattern for time derivatives, e.g. 'fname_derivs_%03d.npy'.

        Notes
        -----
        Each MPI process instantiates its own mpi_pool and loads only the subset
        of trajectories assigned to that rank. The full dataset is distributed
        across the MPI pool.
        """

        if MPI is None:
            raise ModuleNotFoundError(
                "mpi4py is required to use mpi_pool. Install mpi4py or avoid MPI-dependent workflows."
            )

        self.comm = comm                            # MPI communicator
        self.size = self.comm.Get_size()            # Total number of processes
        self.rank = self.comm.Get_rank()            # Id of the current process

        self.n_traj = n_traj                        # Total number of training trajectories
        if self.size > self.n_traj:
            raise ValueError ("You have more MPI processes than trajectories!")
        else:
            if self.rank == 0:
                print("Hello, you are running NiTROM with %d MPI processors."%self.size)
        
        self.my_n_traj = self.n_traj//self.size     # Number of trajectories owned by process self.rank
        self.my_n_traj += 1 if np.mod(self.n_traj,self.size) > self.rank else 0


        # Vectors used for future MPI communications
        self.counts = np.zeros(self.size,dtype=np.int64)    
        self.comm.Allgather([np.asarray([self.my_n_traj]),MPI.INT],[self.counts,MPI.INT])
        self.disps = np.concatenate(([0],np.cumsum(self.counts)[:-1])) 
        
        
        # Load data from file
        self.load_trajectories(fname_traj)
        self.time = np.load(fname_time)
        self.load_weights(kwargs)
        self.load_steady_forcing(kwargs)
        self.load_time_derivatives(kwargs)
        
        
    def load_trajectories(self,fname_traj):
        """
        Load trajectory files for the local MPI rank.

        Parameters
        ----------
        fname_traj : str
            Filename pattern for trajectories assigned to this process.

        Notes
        -----
        The method populates:
          - self.fnames_traj : list of filenames loaded by this rank
          - self.X : numpy array of shape (my_n_traj, N, n_snapshots)
          - self.N, self.n_snapshots : dimensions inferred from the first file
        """
        self.fnames_traj = [fname_traj%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
        X = [np.load(self.fnames_traj[k]) for k in range (self.my_n_traj)]
        self.N, self.n_snapshots = X[0].shape
        self.X = np.zeros((self.my_n_traj, self.N, self.n_snapshots))
        for k in range (self.my_n_traj): self.X[k,] = X[k]
        
    def load_weights(self,kwargs):
        """
        Load per-trajectory weights if provided.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed into __init__; looks for 'fname_weights'.

        Notes
        -----
        If no weights file is provided, weights default to ones.
        """        
        fname_weights = kwargs.get('fname_weights',None)
        self.weights = np.ones(self.my_n_traj)
        if fname_weights != None:
            self.fnames_weights = [fname_weights%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            self.weights = np.asarray([np.load(self.fnames_weights[k]) for k in range (self.my_n_traj)])
            
    def load_steady_forcing(self,kwargs):
        """
        Load steady forcing vectors for each local trajectory if provided.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed into __init__; looks for 'fname_steady_forcing'.

        Notes
        -----
        Populates self.F with shape (N, my_n_traj). If no files are provided,
        self.F remains a zero array.
        """        
        fname_forcing = kwargs.get('fname_steady_forcing',None)
        self.F = np.zeros((self.N,self.my_n_traj))
        if fname_forcing != None:
            self.fnames_forcing = [(fname_forcing)%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            for k in range (self.my_n_traj):  self.F[:,k] = np.load(self.fnames_forcing[k])
    
    def load_time_derivatives(self,kwargs):
        """
        Load precomputed time derivatives for trajectories if provided.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed into __init__; looks for 'fname_derivs'.

        Notes
        -----
        Populates self.dX with shape (my_n_traj, N, n_snapshots) when files exist.
        """        
        fname_deriv = kwargs.get('fname_derivs',None)
        if fname_deriv != None:
            self.fnames_deriv = [fname_deriv%(k+self.disps[self.rank]) for k in range (self.my_n_traj)]
            dX = [np.load(self.fnames_deriv[k]) for k in range (self.my_n_traj)]
            self.dX = np.zeros((self.my_n_traj,self.N,self.n_snapshots))
            for k in range (self.my_n_traj): self.dX[k,] = dX[k]

    @classmethod
    def from_single_trajectory(cls, comm, fname_traj, fname_time,
                               n_segments=None, fname_derivs=None,
                               fname_weights=None):
        """
        Build an mpi_pool from a SINGLE trajectory stored as one 2D matrix.

        The trajectory is provided as a ``(states, times)`` matrix (e.g. x.npy)
        and is split into contiguous, equal-length time segments distributed
        across the MPI ranks (one or more segments per rank). This parallelizes
        a lone trajectory with the very same machinery NiTROM already uses for
        many trajectories: each segment is treated as an independent short
        trajectory whose initial condition is the true FOM state at the segment
        start (a multiple-shooting decomposition). The per-rank contributions
        are summed across ranks inside the cost/gradient, so the result is the
        average error over all segments and snapshots.

        All segments share the same length ``L = n_times // n_segments`` and the
        same time grid, so the downstream ``optimization_objects`` (which slices
        a common ``which_times`` across trajectories) works unchanged. With one
        rank and the default ``n_segments`` this reduces to the original
        single-trajectory behavior.

        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator.
        fname_traj : str
            Path to the ``(states, times)`` state matrix, e.g. 'x.npy'.
        fname_time : str
            Path to the time vector of the full trajectory, e.g. 'time.npy'.
        n_segments : int, optional
            Number of segments to split the trajectory into. Defaults to the
            communicator size (one segment per rank). Must be >= comm size.
        fname_derivs : str, optional
            Path to the ``(states, times)`` velocity matrix, e.g. 'xdot.npy'.
        fname_weights : str, optional
            Path to a scalar (or per-segment) weight applied to every segment.
            Defaults to unit weights.
        """
        if MPI is None:
            raise ModuleNotFoundError(
                "mpi4py is required to use mpi_pool. Install mpi4py or avoid MPI-dependent workflows."
            )

        self = cls.__new__(cls)
        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        n_segments = self.size if n_segments is None else int(n_segments)
        if n_segments < self.size:
            raise ValueError(
                "You have more MPI processes (%d) than trajectory segments (%d)!"
                % (self.size, n_segments))

        # Distribute the segments over ranks exactly like the file-based loader.
        self.n_traj = n_segments
        self.my_n_traj = n_segments // self.size
        self.my_n_traj += 1 if np.mod(n_segments, self.size) > self.rank else 0

        self.counts = np.zeros(self.size, dtype=np.int64)
        self.comm.Allgather([np.asarray([self.my_n_traj]), MPI.INT], [self.counts, MPI.INT])
        self.disps = np.concatenate(([0], np.cumsum(self.counts)[:-1]))

        # Memory-map the full matrices so each rank materializes only its segments.
        Xfull = np.load(fname_traj, mmap_mode='r')
        self.N, n_times = Xfull.shape
        L = n_times // n_segments
        if L == 0:
            raise ValueError(
                "n_segments=%d exceeds the %d available snapshots." % (n_segments, n_times))
        self.n_snapshots = L

        if self.rank == 0:
            print("Hello, you are running NiTROM with %d MPI processors." % self.size)
            print("Single trajectory (%d x %d) split into %d segments of %d snapshots."
                  % (self.N, n_times, n_segments, L))
            dropped = n_times - n_segments * L
            if dropped:
                print("  (dropping %d trailing snapshots so all segments are equal length)" % dropped)

        def _segment(mm, g):
            c0 = g * L
            return np.asarray(mm[:, c0:c0 + L])

        self.X = np.zeros((self.my_n_traj, self.N, L))
        for k in range(self.my_n_traj):
            self.X[k] = _segment(Xfull, k + self.disps[self.rank])
        del Xfull

        if fname_derivs is not None:
            dXfull = np.load(fname_derivs, mmap_mode='r')
            self.dX = np.zeros((self.my_n_traj, self.N, L))
            for k in range(self.my_n_traj):
                self.dX[k] = _segment(dXfull, k + self.disps[self.rank])
            del dXfull

        self.time = np.load(fname_time)[:L]
        self.F = np.zeros((self.N, self.my_n_traj))

        self.weights = np.ones(self.my_n_traj)
        if fname_weights is not None:
            w = np.asarray(np.load(fname_weights)).reshape(-1)
            self.weights = np.full(self.my_n_traj, w[0]) if w.size == 1 else w[
                self.disps[self.rank]:self.disps[self.rank] + self.my_n_traj]

        return self


class optimization_objects:

    def __init__(self,
                 mpi_pool,
                 which_trajs,
                 which_times,
                 leggauss_deg,
                 nsave_rom,
                 poly_comp,
                 hamiltonian=False,
                 **kwargs):
        
        r"""
        Prepare training data information and optimization objects to be passed to the optimizer.

        Parameters
        ----------
        mpi_pool : mpi_pool
            Instance of the mpi_pool class containing distributed data.
        which_trajs : array_like
            Indices selecting trajectories from mpi_pool.X to include in this batch. 
            Useful for stochastic gradient descent.
        which_times : array_like
            Indices selecting time snapshots to include from each trajectory. 
            Useful if we want to start training on short trajectories and then progressively 
            extend the length of the trajectories.
        leggauss_deg : int
            Number of Gauss–Legendre quadrature points for integral approximations. 
            For further details, see Prop. 2.1 in NiTROM arXiv paper
        nsave_rom : int
            Number of ROM snapshots stored between successive FOM snapshots.
        poly_comp : sequence of int
            Polynomial components of the ROM; e.g. [1, 2] for linear and quadratic terms.

            .. math::
            f_r = A_r\hat{z} + B_ru + H_r:\hat{z}\hat{z}^T + L_r:\hat{z}u^T + \ldots

            
        **kwargs : optional
            which_fix : {'fix_bases', 'fix_tensors', 'fix_none'}, default 'fix_none'
                Which quantities to keep fixed during optimization.
            stab_promoting_pen : float, optional
                L2 regularization coefficient for stability-promoting penalty.
            stab_promoting_tf : float, optional
                Final time used by the stability-promoting penalty.
            stab_promoting_ic : array_like, optional
                Initial condition (random) normalized vector used to probe stability penalty.

        Raises
        ------
        ValueError
            If invalid which_fix provided, or if required stability penalty arguments are missing.

        Notes
        -----
        This class slices mpi_pool data according to which_trajs and which_times,
        rescales trajectory weights so the cost measures average error over snapshots
        and trajectories, and generates einsum subscripts for efficient tensor contractions.
        """        
        
        self.X = mpi_pool.X[which_trajs,:,:]      
        self.X = self.X[:,:,which_times]
        self.F = mpi_pool.F[:,which_trajs]
        self.time = mpi_pool.time[which_times]
        self.weights = mpi_pool.weights[which_trajs]

        self.my_n_traj, _, self.n_snapshots = self.X.shape
        self.leggauss_deg = leggauss_deg
        self.nsave_rom = nsave_rom
        self.poly_comp = poly_comp
        self.hamiltonian = hamiltonian
        self.generate_einsum_subscripts()
        
        # Count the total number of trajectories in this batch and
        # scale the weight accordingly so that the cost function measures
        # the average error over snapshots and trajectories. (Notice that 
        # if all trajectories are loaded, then np.sum(counts) = mpi_pool.n_traj)
        counts = np.zeros(mpi_pool.size,dtype=np.int64)
        mpi_pool.comm.Allgather([np.asarray([self.my_n_traj]),MPI.INT],[counts,MPI.INT])
        self.weights *= np.sum(counts)*self.n_snapshots
        
        # Parse the keyword arguments
        self.which_fix = kwargs.get('which_fix','fix_none')
        if self.which_fix not in ['fix_tensors','fix_bases','fix_none']:
            raise ValueError ("which_fix must be fix_none, fix_tensors or fix_bases")

        # Internal step for symplectic ROM integration. ``None`` falls back to
        # one symplectic step per snapshot interval (the legacy behavior).
        self.dt_rom = kwargs.get('dt_rom', None)

        self.l2_pen = kwargs.get('stab_promoting_pen',None)
        self.pen_tf = kwargs.get('stab_promoting_tf',None)
        self.randic = kwargs.get('stab_promoting_ic',None)
        
        if self.l2_pen != None and self.pen_tf == None:
            raise ValueError ("If you provide a value for stab_promoting_pen you \
                              also have to provide a value for stab_promoting_tf")
                              
        if self.l2_pen != None and self.randic == None:
            raise ValueError ("If you provide a value for stab_promoting_pen you \
                              also have to provide a random ic vector of the same \
                              size as the ROM")
                              
        if self.l2_pen != None and 1 not in self.poly_comp:
            raise ValueError ("The penalty is currently implemented for the linear term \
                              in the rom dynamics. You have no linear term.")
                              
        if self.randic != None: 
            self.randic /= np.linalg.norm(self.randic)
            self.randic = self.randic.reshape(-1)
            
        
    
    def generate_einsum_subscripts(self):
        """
            Generates the indices for the einsum evaluation of the 
            right-hand side and the adjoint
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii[:k+1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)
        
        self.einsum_ss = tuple(ss)

    def build_rom(self, Phi, Psi=None, operators=None):
        """
        Instantiate the ROM associated with the active reduction type.
        """
        if self.hamiltonian:
            return HamiltonianROM(operators=operators, poly_comp=self.poly_comp, Phi=Phi)

        if Psi is None:
            raise ValueError("Polynomial reduction requires both Phi and Psi.")
        return PolynomialROM(operators=operators, poly_comp=self.poly_comp, Phi=Phi, Psi=Psi)

    def build_reduction_operators(self, Phi, Psi=None, operators=None):
        """
        Build the projection operators associated with the active reduction type.
        """
        rom = self.build_rom(Phi, Psi=Psi, operators=operators)
        operators_dict = rom.build_projection_operators()
        operators_dict["rom"] = rom
        return operators_dict

    def build_objective(self, rom, fom, mpi_pool):
        """
        Instantiate the optimization objective associated with the active reduction type.
        """
        if self.hamiltonian:
            return HamiltonianReductionObjective(rom=rom, fom=fom, opt_obj=self, mpi_pool=mpi_pool)
        return PolynomialReductionObjective(rom=rom, fom=fom, opt_obj=self, mpi_pool=mpi_pool)
        
    def evaluate_rom_rhs(self, t, z, u, *operators, **kwargs):
        """
        Evaluate  reduced dynamics for solve_ivp.

        Parameters
        ----------
        t : float
        z : np.ndarray
        u : ndarray or callable
        *operators : tuple
            Polynomial operators (A2, A3, ...)
        kwargs : dict
            Optional forcing interpolator etc.
        """
        rom = PolynomialROM(operators=operators, poly_comp=self.poly_comp)
        return rom.evaluate_rom_rhs(t, z, u, *operators, **kwargs)

    def evaluate_rom_adjoint(self,t,z,fq,*operators):
        """
            Function that can be fed into scipys solve_ivp. 
            t:          time instance
            z:          state vector
            fq:         interpolator (from scipy.interpolate) to evaluate the
                        base flow at time t
            operators:  (A2,A3,A4,...)
        """
        rom = PolynomialROM(operators=operators, poly_comp=self.poly_comp)
        return rom.evaluate_rom_adjoint(t, z, fq, *operators)

class BaseROM:

    def __init__(self,
                 operators=None,
                 poly_comp=None,
                 Phi=None,
                 Psi=None):

        self.operators = list(operators) if operators is not None else []
        self.poly_comp = list(poly_comp) if poly_comp is not None else []
        self.Phi = Phi
        self.Psi = Psi
        self.einsum_ss = tuple()

        if self.poly_comp:
            self.generate_einsum_subscripts()

    def generate_einsum_subscripts(self):
        """
        Generates einsum indices for polynomial ROM terms.
        """
        ss = []
        for k in self.poly_comp:
            ssk = ascii[:k + 1]
            ssk = [ssk] + [s for s in ssk[1:]]
            ss.append(ssk)

        self.einsum_ss = tuple(ss)

    def build_projection_operators(self):
        raise NotImplementedError

    def encode(self, x):
        return self.build_projection_operators()["encoder"] @ x

    def reconstruct(self, z):
        return self.build_projection_operators()["decoder"] @ z

    def project(self, x):
        return self.build_projection_operators()["projector"] @ x

    def integrate(self, time, z0, forcing=None):
        raise NotImplementedError


class PolynomialROM(BaseROM):

    def build_projection_operators(self):
        """
        Build the Petrov-Galerkin projection operators associated with (Phi, Psi).
        """
        if self.Phi is None or self.Psi is None:
            raise ValueError("Polynomial ROM projection operators require Phi and Psi.")

        F = np.linalg.inv(self.Psi.T @ self.Phi)
        decoder = self.Phi @ F
        encoder = self.Psi.T
        projector = decoder @ encoder

        return {
            "encoder": encoder,
            "decoder": decoder,
            "projector": projector,
            "F": F,
        }

    def integrate(self, time, z0, forcing=None):
        """
        Integrate the polynomial ROM on the time grid `time`.
        """
        if forcing is None:
            forcing = np.zeros_like(z0)

        sol = solve_ivp(
            self.evaluate_rom_rhs,
            [time[0], time[-1]],
            z0,
            method="RK45",
            t_eval=time,
            args=(forcing,) + tuple(self.operators),
        )
        return sol.y

    def evaluate_rom_rhs(self, t, z, u, *operators, **kwargs):
        """
        Evaluate reduced polynomial dynamics for solve_ivp.
        """
        r = len(z)
        if np.linalg.norm(z) >= 1e4:
            return np.zeros_like(z)

        forcing_interp = kwargs.get('forcing_interp', None)
        f = forcing_interp(t) if forcing_interp is not None else np.zeros(r)
        u_val = u.copy() if hasattr(u, "__len__") else u(t)

        dzdt = u_val + f
        active_operators = operators if len(operators) > 0 else tuple(self.operators)
        for (i, k) in enumerate(self.poly_comp):
            equation = ",".join(self.einsum_ss[i])
            operands = [active_operators[i]] + [z for _ in range(k)]
            dzdt += np.einsum(equation, *operands)

        return dzdt

    def evaluate_rom_adjoint(self, t, z, fq, *operators):
        """
        Evaluate the adjoint of the polynomial ROM.
        """
        if np.linalg.norm(z) >= 1e4:
            return 0.0 * z

        J = np.zeros((len(z), len(z)))
        active_operators = operators if len(operators) > 0 else tuple(self.operators)
        for (i, k) in enumerate(self.poly_comp):
            combs = list(combinations(self.einsum_ss[i][1:], r=k - 1))
            operands = [active_operators[i]] + [fq(t) for _ in range(k - 1)]
            for comb in combs:
                equation = [self.einsum_ss[i][0]] + list(comb)
                equation = ",".join(equation)
                J += np.einsum(equation, *operands)

        return J.T @ z


class HamiltonianROM(BaseROM):

    def __init__(self, 
                 operators=None, 
                 poly_comp=None, 
                 Phi=None, 
                 Psi=None):

        super().__init__(operators=operators, poly_comp=poly_comp, Phi=Phi, Psi=Psi)
        self.r = None
        self.J = None
        self.Jhat = None

        # Coefficients for the six-stage symmetric Runge-Kutta-Nystrom scheme
        self.a  = [0.245298957184271,
                   0.604872665711080,
                   0.5 - (0.245298957184271 + 0.604872665711080),
                   0.5 - (0.245298957184271 + 0.604872665711080),
                   0.604872665711080,
                   0.245298957184271]
        self.b = [0.0829844064174052,
                  0.396309801498368,
                  -0.0390563049223486,
                  1 - 2 * (0.0829844064174052 + 0.396309801498368 + -0.0390563049223486),
                  -0.0390563049223486,
                  0.396309801498368,
                  0.0829844064174052]
        if self.Phi is not None:
            self._update_symplectic_structure()

    def _update_symplectic_structure(self):
        """
        Build the canonical symplectic matrix and reduced operator Jhat.
        """
        ambient_dim, self.r = self.Phi.shape
        if ambient_dim % 2 != 0:
            raise ValueError("Hamiltonian ROM expects an even ambient dimension.")

        half_dim = ambient_dim // 2
        self.J = bmat([[csr_matrix((half_dim, half_dim)), eye(half_dim)],
                       [-eye(half_dim), csr_matrix((half_dim, half_dim))]])
        self.Jhat = self.Phi.T @ self.J @ self.Phi

    def build_projection_operators(self):
        """
        Build the symplectic reduction operators associated with Phi.

        J stays SPARSE and the n x n projector is never formed: for the
        uncompressed FOM (n ~ 1e4+) a dense J or decoder @ encoder is O(n^2)
        memory (GBs) allocated on EVERY cost/gradient call. Everything the
        objectives need is J-matvecs and the thin (n x r) JU / decoder.
        """
        if self.Phi is None:
            raise ValueError("Hamiltonian ROM projection operators require Phi.")

        U = self.Phi
        JU = np.asarray(self.J @ U)                  # thin (n, r); J stays sparse
        M = JU.T @ U
        M_inv = np.linalg.inv(M)
        M_inv_T = M_inv.T
        decoder = U @ M_inv
        encoder = JU.T

        return {
            "encoder": encoder,
            "decoder": decoder,
            "J": self.J,
            "JU": JU,
            "M": M,
            "M_inv": M_inv,
            "M_inv_T": M_inv_T,
            "Jhat": self.Jhat,
        }

    def project(self, x):
        """Symplectic projection decoder @ (encoder @ x), factored so the n x n
        projector matrix is never materialized."""
        ops = self.build_projection_operators()
        return ops["decoder"] @ (ops["encoder"] @ x)

    def integrate(self, time: np.ndarray, x0: np.ndarray, forcing=None, dt=None) -> list[np.ndarray, np.array]:
        """
        Integrate the Hamiltonian ROM on the time grid `time` with a fourth-order
        symplectic method valid for nonseparable Hamiltonians.

        Parameters
        ----------
        time    : array of time points at which to evaluate the solution
        x0      : initial condition
        forcing : array of shape (r, len(time)) containing the forcing term at each time point
        dt      : time step for the internal integration; if None, it will be inferred from the time grid.

        Returns
        -------
        X : array of shape (r, len(time)) containing the ROM state at each time point
        # H : array of shape (len(time),) containing the Hamiltonian evaluated at each time point

        """
        # print(f"[rom.integrate] dt={dt}, len(time)={len(time)}", flush=True)

        if forcing is not None:
            raise NotImplementedError("HamiltonianROM.integrate does not support forcing.")

        if len(time) < 2:
            raise ValueError("Hamiltonian ROM integration requires at least two time points.")

        # Fast path: a purely quadratic Hamiltonian (poly_comp == [1]) has LINEAR
        # dynamics  z' = A z,  A = Jhat (A2 + A2^T),  so the implicit-midpoint solve
        # is the exact Cayley transform -- no Newton, propagator precomputed once.
        # Identical trajectory to the general integrator (to round-off), far cheaper.
        if list(self.poly_comp) == [1]:
            A2 = self.operators[0]
            A_dyn = np.asarray(self.Jhat) @ (A2 + A2.T)
            return linear_symplectic_solve(A_dyn, x0, time, dt=dt)

        X = symplectic_solve(
            rhs=self.evaluate_rom_rhs,
            jacobian=self.evaluate_rom_rhs_jacobian,
            z0=x0,
            t_eval=time,
            dt=dt,
        )

        # # Test whether training diverges because of integrator. 
        # X = solve_ivp(
        #     fun=lambda t, x: self.evaluate_rom_rhs(x),
        #     t_span=(time[0], time[-1]),
        #     y0=x0,
        #     t_eval=time,
        #     method="Radau",
        # ).y

        return X
        
    def compute_hamiltonian(self, x: np.ndarray) -> float:
            r"""
            Evaluate the polynomial Hamiltonian at state x:
            .. math::
                H(x) = 1/2 x^\top A_2 x + 1/3 x^\top A_3 : x x^\top + \ldots
            """

            poly_comp = self.poly_comp
            operators = self.operators
            einsum_ss = self.einsum_ss
            H = 0.0

            for i, k in enumerate(poly_comp):
                term = np.einsum(",".join(einsum_ss[i]), operators[i], *([x] * k))/k
                H += np.dot(x, term)
                # aggiungere torch.autograd
            return H

    def compute_hamiltonian_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the polynomial Hamiltonian at state x.
        """
        grad_H = np.zeros_like(x)

        for operator, k in zip(self.operators, self.poly_comp):
            tensor_ss = ascii[:k + 1]

            for slot in range(k + 1):
                kept_index = tensor_ss[slot]
                contracted_indices = [tensor_ss[j] for j in range(k + 1) if j != slot]
                equation_terms = [tensor_ss] + contracted_indices
                equation = ",".join(equation_terms) + f"->{kept_index}"
                operands = [operator] + [x] * len(contracted_indices)
                grad_H += np.einsum(equation, *operands)

        return grad_H

    def compute_hamiltonian_hessian(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Hessian of the polynomial Hamiltonian at state x.
        """
        hess_H = np.zeros((len(x), len(x)), dtype=x.dtype)

        for operator, k in zip(self.operators, self.poly_comp):
            tensor_ss = ascii[:k + 1]

            for slot_a in range(k + 1):
                kept_a = tensor_ss[slot_a]
                for slot_b in range(k + 1):
                    if slot_b == slot_a:
                        continue

                    kept_b = tensor_ss[slot_b]
                    contracted = [
                        tensor_ss[j]
                        for j in range(k + 1)
                        if j not in (slot_a, slot_b)
                    ]
                    equation_terms = [tensor_ss] + contracted
                    equation = ",".join(equation_terms) + f"->{kept_a}{kept_b}"
                    operands = [operator] + [x] * len(contracted)
                    hess_H += np.einsum(equation, *operands)

                    ## Hessian of z^\top H : zz^\top : LOGIC
                    # if len(operators) > 1:
                    #     # T = H + H^{(1,2)} + H^{(1,3)}, call H "A2"
                    #     A2 = operators[1]
                    #     T = A2 + np.einsum('ijk->jik', A2) + np.einsum('ikj->kji', A2)
                    #     # (i,j) entry: sum_k T_ijk z_k + sum_k T_ikj z_k
                    #     hessian_hamiltonian += np.einsum('ijk,k->ij', T, z) + np.einsum('ikj,k->ij', T, z)
        
        return hess_H
    
    def evaluate_rom_rhs(self, x: np.ndarray) -> np.ndarray:
        r"""
        Evaluate the right-hand side of the Hamiltonian ROM at state x:
        ..math::
            \dot{z} = hat_{J} \nabla_z H(z)
        where
        ..math::
        hat_{J} = Phi^T J Phi
        and
        ..math::
        H(z) = z^\top A_2 z + z^\top A_3 : z z^\top + \ldots

        Parameters
        ----------
        t : float
        """
        t0 = time_module.perf_counter()
        if len(x) % 2 != 0:
            raise ValueError("Hamiltonian ROM expects even dimension z = [q,p].")
        
        nabla_H = self.compute_hamiltonian_gradient(x)
        dxdt = self.Jhat @ nabla_H
        record_runtime("ham_rom_rhs_eval", time_module.perf_counter() - t0)
        
        return dxdt

    def evaluate_rom_rhs_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Jacobian of the Hamiltonian ROM vector field.
        """
        if len(x) % 2 != 0:
            raise ValueError("Hamiltonian ROM expects even dimension z = [q,p].")

        hess_H = self.compute_hamiltonian_hessian(x)
        return self.Jhat @ hess_H


    def evaluate_rom_adjoint(self, t, lam, fz, *operators):
        # print("Evaluating Hamiltonian ROM adjoint at time t =", t)
        if np.linalg.norm(lam) >= 1e4:
            return 0.0 * lam

        z = fz(t)
        hessian_hamiltonian = HamiltonianROM(
            operators=operators if len(operators) > 0 else self.operators,
            poly_comp=self.poly_comp,
            Phi=self.Phi,
        ).compute_hamiltonian_hessian(z)

        # adjoint ODE: -d lambda/dt = H2 @ Jhat^T @ lambda
        return -(hessian_hamiltonian @ self.Jhat.T @ lam)

class BaseReductionObjective:

    def __init__(self, rom, fom, opt_obj, mpi_pool):
        self.rom = rom
        self.fom = fom
        self.opt_obj = opt_obj
        self.mpi_pool = mpi_pool
        self.tlg, self.wlg = np.polynomial.legendre.leggauss(opt_obj.leggauss_deg)
        self.wlg = np.asarray(self.wlg)

    def trajectory_cost(self, traj_idx):
        raise NotImplementedError

    def trajectory_gradient(self, traj_idx):
        raise NotImplementedError


class PolynomialReductionObjective(BaseReductionObjective):

    def __init__(self, rom, fom, opt_obj, mpi_pool):
        super().__init__(rom=rom, fom=fom, opt_obj=opt_obj, mpi_pool=mpi_pool)
        ops = rom.build_projection_operators()
        self.decoder = ops["decoder"]
        self.F = ops["F"]
        self.r = self.decoder.shape[1]
        self.n = self.decoder.shape[0]

    def trajectory_cost(self, traj_idx):
        Xk = self.opt_obj.X[traj_idx]
        z0 = self.rom.encode(Xk[:, 0])
        u = self.rom.encode(self.opt_obj.F[:, traj_idx])
        Z = self.rom.integrate(self.opt_obj.time, z0, forcing=u)
        error = self.fom.compute_output(Xk) - self.fom.compute_output(self.rom.reconstruct(Z))
        return (1.0 / self.opt_obj.weights[traj_idx]) * np.trace(error.T @ error)

    def trajectory_gradient(self, traj_idx):
        Xk = self.opt_obj.X[traj_idx]
        alpha = self.opt_obj.weights[traj_idx]
        z0 = self.rom.encode(Xk[:, 0])
        u = self.rom.encode(self.opt_obj.F[:, traj_idx])
        Z = self.rom.integrate(self.opt_obj.time, z0, forcing=u)
        e = self.fom.compute_output(Xk) - self.fom.compute_output(self.rom.reconstruct(Z))

        grad_Phi = np.zeros_like(self.rom.Phi)
        grad_Psi = np.zeros_like(self.rom.Psi)
        grad_tensors = [np.zeros_like(op) for op in self.rom.operators]
        lam_j_0 = np.zeros(self.r)
        Int_lambda = np.zeros(self.r)

        for j in range(self.opt_obj.n_snapshots - 1):
            ej = e[:, self.opt_obj.n_snapshots - j - 1]
            zj = Z[:, self.opt_obj.n_snapshots - j - 1]
            Ctej = self.fom.compute_output_derivative(self.decoder @ zj).T @ ej

            grad_Psi += (2 / alpha) * np.einsum('i,j', self.decoder @ zj, self.decoder.T @ Ctej)
            grad_Phi += -(2 / alpha) * np.einsum(
                'i,j',
                Ctej - self.rom.Psi @ (self.decoder.T @ Ctej),
                self.F @ zj,
            )

            id1 = self.opt_obj.n_snapshots - 1 - j
            id0 = id1 - 1
            tf_j = self.opt_obj.time[id1]
            t0_j = self.opt_obj.time[id0]
            z0_j = Z[:, id0]

            time_rom_j = np.linspace(t0_j, tf_j, num=self.opt_obj.nsave_rom, endpoint=True)
            time_rom_j_back = time_rom_j[::-1]                                   # reverse time for backward integration
            Z_j = self.rom.integrate(time_rom_j, z0_j, forcing=u)
            Z_j = np.fliplr(Z_j)

            fZ = interp1d(time_rom_j, Z_j, kind='linear', fill_value='extrapolate')

            lam_j_0 += (2 / alpha) * self.decoder.T @ Ctej

            sol_lam = solve_ivp(
                self.rom.evaluate_rom_adjoint,
                [tf_j, t0_j],
                lam_j_0,
                method='RK45',
                t_eval=time_rom_j,
                args=(fZ,) + tuple(self.rom.operators),
            )
            Lam = np.fliplr(sol_lam.y)
            lam_j_0 = Lam[:, 0]
            Z_j = np.fliplr(Z_j)

            a = (tf_j - t0_j) / 2
            b = (tf_j + t0_j) / 2
            time_j_lg = a * self.tlg + b

            fZ = interp1d(time_rom_j, Z_j, kind='linear', fill_value='extrapolate')
            fL = interp1d(time_rom_j, Lam, kind='linear', fill_value='extrapolate')
            Z_j_lg = fZ(time_j_lg)
            Lam_lg = fL(time_j_lg)

            for i in range(self.opt_obj.leggauss_deg):
                Int_lambda += a * self.wlg[i] * Lam_lg[:, i]
                for count, p in enumerate(self.opt_obj.poly_comp):
                    equation = ','.join(ascii[:p + 1])
                    operands = [Lam_lg[:, i]] + [Z_j_lg[:, i] for _ in range(p)]
                    grad_tensors[count] -= a * self.wlg[i] * np.einsum(equation, *operands)

# are z and lambda evaluated at the same times?
        ej, zj = e[:, 0], Z[:, 0]
        Ctej = self.fom.compute_output_derivative(self.decoder @ zj).T @ ej
        grad_Psi += (2 / alpha) * np.einsum('i,j', self.decoder @ zj, self.decoder.T @ Ctej) \
                    - np.einsum('i,j', Xk[:, 0], lam_j_0) \
                    - np.einsum('i,j', self.opt_obj.F[:, traj_idx], Int_lambda)
        grad_Phi += -(2 / alpha) * np.einsum(
            'i,j',
            Ctej - self.rom.Psi @ (self.decoder.T @ Ctej),
            self.F @ zj,
        )

        return {
            "grad_Phi": grad_Phi,
            "grad_Psi": grad_Psi,
            "grad_tensors": grad_tensors,
        }


class HamiltonianReductionObjective(BaseReductionObjective):

    def __init__(self, rom, fom, opt_obj, mpi_pool):
        super().__init__(rom=rom, fom=fom, opt_obj=opt_obj, mpi_pool=mpi_pool)
        ops = rom.build_projection_operators()
        self.J = ops["J"]
        self.JU = ops["JU"]
        self.M_inv = ops["M_inv"]
        self.M_inv_T = ops["M_inv_T"]
        self.D = ops["decoder"]
        self.Jhat = ops["Jhat"]
        self.U = rom.Phi
        self.r = self.D.shape[1]
        self.n = self.D.shape[0]

    # ── Linear fast path (poly_comp == [1]) ─────────────────────────────────
    # A quadratic Hamiltonian has LINEAR dynamics  z' = A z,  A = Jhat (A2+A2^T)
    # constant within one cost/gradient call, so the symplectic march is a
    # constant matrix: states (and adjoints) anywhere are obtained with
    # precomputed propagators instead of sub-stepping at dt_rom (forward) and
    # RK45 (adjoint). Over one snapshot interval of length ``gap``:
    #
    #   z(t + gap)  = M_gap @ z(t)          M_gap = linear_propagator(A, gap)
    #   z(t + d_i)  = N_i   @ z(t)          d_i   = Gauss-node offsets from t
    #   lam(t)      = M_gap.T @ lam(t+gap)  adjoint lam' = -A^T lam: the Cayley
    #   lam(t + d_i)= Q_i.T   @ lam(t+gap)  step commutes with transposition,
    #                                       so the discrete backward-adjoint map
    #                                       is exactly the forward map transposed
    #                                       (Q_i propagates node i -> interval end).
    # Same discrete trajectory as the step-by-step march (products of the same
    # Cayley factors), so nothing changes numerically -- it only removes the
    # per-step Python loop, the spline at the Gauss nodes, and the RK45 adjoint.

    def _use_linear_fast_path(self):
        return list(self.rom.poly_comp) == [1]

    def _linear_dynamics(self):
        """Constant reduced dynamics A = Jhat (A2 + A2^T) of the quadratic H."""
        A2 = self.rom.operators[0]
        return np.asarray(self.Jhat) @ (A2 + A2.T)

    def _march_snapshots(self, A, z0):
        """Snapshot states Z[:, k] = z(t_k) by one compounded propagator per gap
        (one propagator total on a uniform grid; cached per distinct gap)."""
        time = self.opt_obj.time
        dt = self.opt_obj.dt_rom
        Z = np.empty((self.r, len(time)))
        Z[:, 0] = z0
        maps = {}
        for k in range(1, len(time)):
            gap = float(time[k] - time[k - 1])
            key = round(gap, 15)
            if key not in maps:
                maps[key] = linear_propagator(A, gap, dt)
            Z[:, k] = maps[key] @ Z[:, k - 1]
        return Z

    def _interval_maps(self, A, gap):
        """(M_gap, N, Q) for one snapshot interval: full-gap propagator, and the
        node propagators N_i (t -> node i) and Q_i (node i -> t + gap) at the
        Gauss-Legendre nodes d_i = gap (tlg_i + 1) / 2."""
        dt = self.opt_obj.dt_rom
        M_gap = linear_propagator(A, gap, dt)
        offsets = 0.5 * gap * (self.tlg + 1.0)
        N = [linear_propagator(A, d, dt) for d in offsets]
        Q = [linear_propagator(A, gap - d, dt) for d in offsets]
        return M_gap, N, Q

    def trajectory_cost(self, traj_idx):
        Xk = self.opt_obj.X[traj_idx]
        z0 = self.rom.encode(Xk[:, 0])
        if self._use_linear_fast_path():
            Z = self._march_snapshots(self._linear_dynamics(), z0)
        else:
            Z = self.rom.integrate(self.opt_obj.time, z0, dt=self.opt_obj.dt_rom)
        error = Xk - self.rom.reconstruct(Z)
        alpha  =  self.opt_obj.weights[traj_idx]
        cost = float((1.0 / alpha) * np.trace(error.T @ error))

        # NO SCALING BY NUMBER OF SNAPSHOTS OR TRAJECTORIES!!!!!!!!!!!!
        # WEIGHTS SHOULD SCALE ACCORDING TO THE ENERGY CONTENT OF THE TRAJECTORY
        # SO ALPHA SHOULD NOT BE ENOUGH.

        return cost

    def trajectory_gradient(self, traj_idx):
        # print(">>> HamiltonianReductionObjective.trajectory_gradient called")
        
        Xk = self.opt_obj.X[traj_idx]
        alpha = self.opt_obj.weights[traj_idx]    
        z0 = self.rom.encode(Xk[:, 0])

        # print("Weights cost function", alpha)

        linear = self._use_linear_fast_path()
        if linear:
            A_lin = self._linear_dynamics()
            interval_maps = {}          # gap -> (M_gap, N, Q); one entry on a uniform grid
            Z = self._march_snapshots(A_lin, z0)
        else:
            Z = self.rom.integrate(self.opt_obj.time, z0, dt=self.opt_obj.dt_rom)
        E = Xk - self.rom.reconstruct(Z)

        grad_Phi = np.zeros_like(self.rom.Phi)
        # grad_Psi = np.zeros_like(self.rom.Phi)
        grad_tensors = [np.zeros_like(op) for op in self.rom.operators]
        lam_j_0 = np.zeros(self.r)

        grad_dynamic = np.zeros_like(self.rom.Phi)
        grad_terminal= np.zeros_like(self.rom.Phi)

        # Static term, vectorized over snapshots: sum_k outer(a_k, b_k) = A B^T
        # with a_k / b_k as columns. Same sums as the per-snapshot outer-product
        # loop, but as three thin (n x ns)(ns x r) matmuls -- the loop version
        # allocates an (n x r) outer product per snapshot, which dominates the
        # gradient at full FOM dimension (n ~ 1e4+).
        MZ    = self.M_inv @ Z                                   # (r, ns)
        DTE   = self.D.T @ E                                     # (r, ns)
        term1 = E @ MZ.T
        term2 = np.asarray(self.J.T @ (self.D @ Z)) @ DTE.T
        term3 = (self.JU @ DTE) @ MZ.T
        grad_static = -(2 / alpha) * (term1 - term2 - term3)

        # Dynamic term: the Gauss-node loop below only accumulates the r x r
        # quadrature sum  S1 = sum_j,i  w_i * outer(lam_i, grad H(z_i)); the
        # expensive (n x r) products J^T U S1 + J U S1^T are applied ONCE after
        # the loop (note sum_i outer(g_i, l_i) = S1^T).
        S1 = np.zeros((self.r, self.r))

        for j in range(self.opt_obj.n_snapshots - 1):
            id1 = self.opt_obj.n_snapshots - 1 - j
            id0 = id1 - 1
            ej = E[:, id1]
            zj = Z[:, id1]

            tf_j = self.opt_obj.time[id1]
            t0_j = self.opt_obj.time[id0]
            z0_j = Z[:, id0]
            a = (tf_j - t0_j) / 2                     # Gauss-Legendre half-width

            lam_j_0 += (2 / alpha) * self.D.T @ ej    # cost injection at t = tf_j

            if linear:
                # States and adjoints at the Gauss nodes by EXACT propagation
                # (see the fast-path note above); lam_j_0 enters as lam(tf_j)
                # and leaves as lam(t0_j). No sub-stepping, spline, or RK45.
                gap = float(tf_j - t0_j)
                key = round(gap, 15)
                if key not in interval_maps:
                    interval_maps[key] = self._interval_maps(A_lin, gap)
                M_gap, N, Q = interval_maps[key]
                Z_j_lg = np.column_stack([Ni @ z0_j for Ni in N])
                Lam_lg = np.column_stack([Qi.T @ lam_j_0 for Qi in Q])
                lam_j_0 = M_gap.T @ lam_j_0
            else:
                time_rom_j = np.linspace(t0_j, tf_j, num=self.opt_obj.nsave_rom, endpoint=True) # forward time integration grid
                Z_j = self.rom.integrate(time_rom_j, z0_j, dt=self.opt_obj.dt_rom)

                fZ = interp1d(time_rom_j, Z_j, kind='cubic', fill_value='extrapolate')

                sol_lam = solve_ivp(
                    self.rom.evaluate_rom_adjoint,
                    [tf_j, t0_j],
                    lam_j_0,
                    method='RK45',
                    t_eval=time_rom_j[::-1],
                    args=(fZ,) + tuple(self.rom.operators),
                )

                Lam = np.fliplr(sol_lam.y)
                lam_j_0 = Lam[:, 0]

                time_j_lg = a * self.tlg + (tf_j + t0_j) / 2

                fL = interp1d(time_rom_j, Lam, kind='cubic', fill_value='extrapolate')

                Z_j_lg = fZ(time_j_lg)
                Lam_lg = fL(time_j_lg)

            for i in range(self.opt_obj.leggauss_deg):

                zi        = Z_j_lg[:, i] # Re-define z inside leggaus? YES!
                li        = Lam_lg[:, i]
                weight    = a * self.wlg[i]
                nabla_H_j = self.rom.compute_hamiltonian_gradient(zi)

                S1 += weight * np.outer(li, nabla_H_j)   # r x r; applied after the loop


                if 1 in self.opt_obj.poly_comp:
                    idx_A2 = self.opt_obj.poly_comp.index(1)
                    # The bracket below is already the full two-term A2 gradient
                    # (it accounts for the A2 + A2^T symmetry in grad H), so the
                    # prefactor is -1, not -2 (the 2 double-counted -> grad was 2x).
                    grad_tensors[idx_A2] +=  -1 * weight * (
                       (np.outer(self.Jhat.T @ li, zi)
                        + np.outer(zi, li) @ self.Jhat))

                if 2 in self.opt_obj.poly_comp:
                    idx_A3 = self.opt_obj.poly_comp.index(2)
                    JhatT_lambda = self.Jhat.T @ li
                    grad_tensors[idx_A3] += -weight * (
                        np.einsum('i,j,k->ijk', JhatT_lambda, zi, zi)
                        + np.einsum('i,j,k->ijk', zj, JhatT_lambda, zi)
                        + np.einsum('i,j,k->ijk', zj, zj, JhatT_lambda)
                    )


        # Apply the accumulated quadrature sum (see S1 note above): two thin
        # (n x r)(r x r) products + sparse-J matmuls, once per trajectory.
        grad_dynamic = -(np.asarray(self.J.T @ (self.U @ S1))
                         + np.asarray(self.J @ (self.U @ S1.T)))

        lam_j_0 += (2 / alpha) * self.D.T @ E[:, 0]
        # grad_terminal += -(1 / alpha) * np.outer(self.J.T @ Xk[:, 0], lam_j_0)
        grad_terminal += - np.outer(self.J.T @ Xk[:, 0], lam_j_0)

        grad_Phi = grad_static + grad_dynamic + grad_terminal


        return {
                "grad_Phi": grad_Phi,
                "grad_Phi_static": grad_static,
                "grad_Phi_dynamic": grad_dynamic,
                "grad_Phi_terminal": grad_terminal,
                # "grad_Psi": grad_Psi,
                "grad_tensors": grad_tensors,
            }


    
