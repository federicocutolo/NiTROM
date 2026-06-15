import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from string import ascii_lowercase as ascii
import pymanopt
import autograd.numpy as anp
import time as tlib

from .classes import debug_enabled, format_runtime_profile, reset_runtime_profile, timing_enabled


def create_objective_and_gradient(manifold,opt_obj,mpi_pool,fom):    
    """
    opt_obj:        instance of class "optimization_objects" in file "nitrom_classes.py"
    mpi_pool:       instance of the class "mpi_pool" in file "nitrom_classes.py"
    fom:            instance of the full-order model class 
    """
    euclidean_hessian = None

    @pymanopt.function.numpy(manifold)
    def cost(*params):
        """ 
            Evaluate the cost function 

            For Hamiltonian:    params = (Phi, A2, A3, ...)
            For Polynomial:     params = (Phi, Psi, A2, A3, ...)
        """
        reset_runtime_profile()
        eval_t0 = tlib.perf_counter()

        if opt_obj.hamiltonian:
            Phi = params[0]
            tensors = params[1:]
        else:
            Phi, Psi = params[0], params[1]
            tensors = params[2:]
        rom = opt_obj.build_rom(Phi, None if opt_obj.hamiltonian else Psi, operators=tensors)
        objective = opt_obj.build_objective(rom, fom, mpi_pool)
        J = 0.0

        for k in range (mpi_pool.my_n_traj): 
            J += objective.trajectory_cost(k)
        
        if opt_obj.l2_pen != None and mpi_pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)    # index of the linear tensor
            time_pen = np.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t,z: tensors[idx]@z if np.linalg.norm(z) < 1e4 else 0*z,\
                           [0,time_pen[-1]],opt_obj.randic,method='RK45',t_eval=time_pen)).y
            J += opt_obj.l2_pen*np.dot(Z[:,-1],Z[:,-1])
            
        gathered_J = mpi_pool.comm.allgather(
            {
                "rank": mpi_pool.rank,
                "type": str(type(J)),
                "shape": np.asarray(J).shape,
                "value": J,
            }
        )
        if debug_enabled() and mpi_pool.rank == 0:
            print("[debug] gathered local costs:", gathered_J)
        J = np.sum(np.asarray([item["value"] for item in gathered_J]))

        if timing_enabled() and mpi_pool.rank == 0:
            print(
                "[timing] cost eval: "
                f"{tlib.perf_counter() - eval_t0:.3e}s total | "
                f"{format_runtime_profile()}"
            )
        # alpha = opt_obj.weights
        # print("Weights cost function", alpha)
        return J

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(*params): 

        """ 
            Evaluate the euclidean gradient of the cost function with respect to the parameters
            For Hamiltonian:    params = (Phi, A2, A3, ...)
            For Polynomial:     params = (Phi, Psi, A2, A3, ...)
        """
        reset_runtime_profile()
        eval_t0 = tlib.perf_counter()

        if opt_obj.hamiltonian:
            Phi = params[0]
            tensors = params[1:]
        else:
            Phi, Psi = params[0], params[1]
            tensors = params[2:]
        rom = opt_obj.build_rom(Phi, None if opt_obj.hamiltonian else Psi, operators=tensors)
        objective = opt_obj.build_objective(rom, fom, mpi_pool)
        
        n, r = Phi.shape
        grad_Phi = np.zeros((n,r))
        grad_Psi = np.zeros((n,r))
        grad_tensors = [0]*len(tensors)
        
        for k in range (mpi_pool.my_n_traj):
            traj_grad = objective.trajectory_gradient(k)
            grad_Phi += traj_grad["grad_Phi"]
            if not opt_obj.hamiltonian:
                grad_Psi += traj_grad["grad_Psi"]
            for idx in range(len(grad_tensors)):
                grad_tensors[idx] += traj_grad["grad_tensors"][idx]

        if opt_obj.l2_pen != None and mpi_pool.rank == 0:
            
            idx = opt_obj.poly_comp.index(1)    # index of the linear tensor
            
            A = tensors[idx]
            
            time_pen = np.linspace(0,opt_obj.pen_tf,opt_obj.n_snapshots*opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t,z: A@z if np.linalg.norm(z) < 1e4 else 0*z,\
                           [0,time_pen[-1]],opt_obj.randic,method='RK45',t_eval=time_pen)).y
            Mu = (solve_ivp(lambda t,z: A.T@z if np.linalg.norm(z) < 1e4 else 0*z,\
                           [0,time_pen[-1]],-2*opt_obj.l2_pen*Z[:,-1],method='RK45',t_eval=time_pen)).y
            Mu = np.fliplr(Mu)
            
            for k in range (opt_obj.n_snapshots - 1):
                k0, k1 = k*opt_obj.nsave_rom, (k+1)*opt_obj.nsave_rom
                fZ = sp.interpolate.interp1d(time_pen[k0:k1],Z[:,k0:k1],kind='linear',fill_value='extrapolate')
                fMu = sp.interpolate.interp1d(time_pen[k0:k1],Mu[:,k0:k1],kind='linear',fill_value='extrapolate')
                a = (time_pen[k1-1] - time_pen[k0])/2
                b = (time_pen[k1-1] + time_pen[k0])/2
                time_k_lg = a*tlg + b
                Zk = fZ(time_k_lg)
                Muk = fMu(time_k_lg)
                for i in range (opt_obj.leggauss_deg):
                    grad_tensors[idx] += -a*wlg[i]*np.einsum('i,j',Muk[:,i],Zk[:,i])

        if opt_obj.which_fix == 'fix_bases':
            grad_Phi *= 0.0
            if not opt_obj.hamiltonian:
                grad_Psi *= 0.0
            for k in range (len(grad_tensors)):
                grad_tensors[k] = sum(mpi_pool.comm.allgather(grad_tensors[k]))

        elif opt_obj.which_fix == 'fix_tensors':    
            for k in range (len(grad_tensors)): grad_tensors[k] *= 0.0
            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            if not opt_obj.hamiltonian:
                grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))

        else: 
            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            if not opt_obj.hamiltonian:
                grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))
            for k in range (len(grad_tensors)):
                grad_tensors[k] = sum(mpi_pool.comm.allgather(grad_tensors[k]))

        if timing_enabled() and mpi_pool.rank == 0:
            print(
                "[timing] grad eval: "
                f"{tlib.perf_counter() - eval_t0:.3e}s total | "
                f"{format_runtime_profile()}"
            )

        if opt_obj.hamiltonian:
            return grad_Phi, *grad_tensors
        else:
            return grad_Phi, grad_Psi, *grad_tensors
    
    return cost, euclidean_gradient, euclidean_hessian


def check_gradient_using_finite_difference(M,Phi,Psi,A2,A3,A4,opt_obj,mpi_pool,fom,eps):
    """Create objective and gradient for Hamiltonian systems (Phi, A2, A3, ...)"""
    
    @pymanopt.function.numpy(manifold)
    def cost_ham(Phi, *tensors):
        """Evaluate cost for Hamiltonian ROM: params = (Phi, A2, A3, ...)"""
        reset_runtime_profile()
        eval_t0 = tlib.perf_counter()

        rom = opt_obj.build_rom(Phi, None, operators=tensors)
        objective = opt_obj.build_objective(rom, fom, mpi_pool)
        J = 0.0

        for k in range(mpi_pool.my_n_traj): 
            J += objective.trajectory_cost(k)
        
        if opt_obj.l2_pen is not None and mpi_pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)
            time_pen = np.linspace(0, opt_obj.pen_tf, opt_obj.n_snapshots * opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t, z: tensors[idx] @ z if np.linalg.norm(z) < 1e4 else 0*z,
                           [0, time_pen[-1]], opt_obj.randic, method='RK45', t_eval=time_pen)).y
            J += opt_obj.l2_pen * np.dot(Z[:, -1], Z[:, -1])
            
        gathered_J = mpi_pool.comm.allgather({
            "rank": mpi_pool.rank,
            "type": str(type(J)),
            "shape": np.asarray(J).shape,
            "value": J,
        })
        if debug_enabled() and mpi_pool.rank == 0:
            print("[debug] gathered local costs:", gathered_J)
        J = np.sum(np.asarray([item["value"] for item in gathered_J]))

        if timing_enabled() and mpi_pool.rank == 0:
            print(
                "[timing] cost eval: "
                f"{tlib.perf_counter() - eval_t0:.3e}s total | "
                f"{format_runtime_profile()}"
            )
        return J

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient_ham(Phi, *tensors):
        """Evaluate gradient for Hamiltonian ROM: params = (Phi, A2, A3, ...)"""
        reset_runtime_profile()
        eval_t0 = tlib.perf_counter()

        rom = opt_obj.build_rom(Phi, None, operators=tensors)
        objective = opt_obj.build_objective(rom, fom, mpi_pool)
        
        n, r = Phi.shape
        grad_Phi = np.zeros((n, r))
        grad_tensors = [0] * len(tensors)
        
        for k in range(mpi_pool.my_n_traj):
            traj_grad = objective.trajectory_gradient(k)
            grad_Phi += traj_grad["grad_Phi"]
            for idx in range(len(grad_tensors)):
                grad_tensors[idx] += traj_grad["grad_tensors"][idx]

        if opt_obj.l2_pen is not None and mpi_pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)
            A = tensors[idx]
            time_pen = np.linspace(0, opt_obj.pen_tf, opt_obj.n_snapshots * opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t, z: A @ z if np.linalg.norm(z) < 1e4 else 0*z,
                           [0, time_pen[-1]], opt_obj.randic, method='RK45', t_eval=time_pen)).y
            Mu = (solve_ivp(lambda t, z: A.T @ z if np.linalg.norm(z) < 1e4 else 0*z,
                           [0, time_pen[-1]], -2*opt_obj.l2_pen*Z[:, -1], method='RK45', t_eval=time_pen)).y
            Mu = np.fliplr(Mu)
            
            for k in range(opt_obj.n_snapshots - 1):
                k0, k1 = k * opt_obj.nsave_rom, (k+1) * opt_obj.nsave_rom
                fZ = sp.interpolate.interp1d(time_pen[k0:k1], Z[:, k0:k1], kind='linear', fill_value='extrapolate')
                fMu = sp.interpolate.interp1d(time_pen[k0:k1], Mu[:, k0:k1], kind='linear', fill_value='extrapolate')
                a = (time_pen[k1-1] - time_pen[k0]) / 2
                b = (time_pen[k1-1] + time_pen[k0]) / 2
                time_k_lg = a * tlg + b
                Zk = fZ(time_k_lg)
                Muk = fMu(time_k_lg)
                for i in range(opt_obj.leggauss_deg):
                    grad_tensors[idx] += -a * wlg[i] * np.einsum('i,j', Muk[:, i], Zk[:, i])

        if opt_obj.which_fix == 'fix_bases':
            grad_Phi *= 0.0
            for k in range(len(grad_tensors)):
                grad_tensors[k] = sum(mpi_pool.comm.allgather(grad_tensors[k]))
        elif opt_obj.which_fix == 'fix_tensors':    
            for k in range(len(grad_tensors)):
                grad_tensors[k] *= 0.0
            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
        else:
            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            for k in range(len(grad_tensors)):
                grad_tensors[k] = sum(mpi_pool.comm.allgather(grad_tensors[k]))

        if timing_enabled() and mpi_pool.rank == 0:
            print(
                "[timing] grad eval: "
                f"{tlib.perf_counter() - eval_t0:.3e}s total | "
                f"{format_runtime_profile()}"
            )
        return grad_Phi, *grad_tensors
    
    return cost_ham, euclidean_gradient_ham, None


def _create_polynomial_objective_and_gradient(manifold, opt_obj, mpi_pool, fom):
    """Create objective and gradient for Polynomial systems (Phi, Psi, A2, A3, ...)"""
    
    @pymanopt.function.numpy(manifold)
    def cost_poly(Phi, Psi, *tensors):
        """Evaluate cost for Polynomial ROM: params = (Phi, Psi, A2, A3, ...)"""
        reset_runtime_profile()
        eval_t0 = tlib.perf_counter()

        rom = opt_obj.build_rom(Phi, Psi, operators=tensors)
        objective = opt_obj.build_objective(rom, fom, mpi_pool)
        J = 0.0

        for k in range(mpi_pool.my_n_traj): 
            J += objective.trajectory_cost(k)
        
        if opt_obj.l2_pen is not None and mpi_pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)
            time_pen = np.linspace(0, opt_obj.pen_tf, opt_obj.n_snapshots * opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t, z: tensors[idx] @ z if np.linalg.norm(z) < 1e4 else 0*z,
                           [0, time_pen[-1]], opt_obj.randic, method='RK45', t_eval=time_pen)).y
            J += opt_obj.l2_pen * np.dot(Z[:, -1], Z[:, -1])
            
        gathered_J = mpi_pool.comm.allgather({
            "rank": mpi_pool.rank,
            "type": str(type(J)),
            "shape": np.asarray(J).shape,
            "value": J,
        })
        if debug_enabled() and mpi_pool.rank == 0:
            print("[debug] gathered local costs:", gathered_J)
        J = np.sum(np.asarray([item["value"] for item in gathered_J]))

        if timing_enabled() and mpi_pool.rank == 0:
            print(
                "[timing] cost eval: "
                f"{tlib.perf_counter() - eval_t0:.3e}s total | "
                f"{format_runtime_profile()}"
            )
        return J

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient_poly(Phi, Psi, *tensors):
        """Evaluate gradient for Polynomial ROM: params = (Phi, Psi, A2, A3, ...)"""
        reset_runtime_profile()
        eval_t0 = tlib.perf_counter()

        rom = opt_obj.build_rom(Phi, Psi, operators=tensors)
        objective = opt_obj.build_objective(rom, fom, mpi_pool)
        
        n, r = Phi.shape
        grad_Phi = np.zeros((n, r))
        grad_Psi = np.zeros((n, r))
        grad_tensors = [0] * len(tensors)
        
        for k in range(mpi_pool.my_n_traj):
            traj_grad = objective.trajectory_gradient(k)
            grad_Phi += traj_grad["grad_Phi"]
            grad_Psi += traj_grad["grad_Psi"]
            for idx in range(len(grad_tensors)):
                grad_tensors[idx] += traj_grad["grad_tensors"][idx]

        if opt_obj.l2_pen is not None and mpi_pool.rank == 0:
            idx = opt_obj.poly_comp.index(1)
            A = tensors[idx]
            time_pen = np.linspace(0, opt_obj.pen_tf, opt_obj.n_snapshots * opt_obj.nsave_rom)
            Z = (solve_ivp(lambda t, z: A @ z if np.linalg.norm(z) < 1e4 else 0*z,
                           [0, time_pen[-1]], opt_obj.randic, method='RK45', t_eval=time_pen)).y
            Mu = (solve_ivp(lambda t, z: A.T @ z if np.linalg.norm(z) < 1e4 else 0*z,
                           [0, time_pen[-1]], -2*opt_obj.l2_pen*Z[:, -1], method='RK45', t_eval=time_pen)).y
            Mu = np.fliplr(Mu)
            
            for k in range(opt_obj.n_snapshots - 1):
                k0, k1 = k * opt_obj.nsave_rom, (k+1) * opt_obj.nsave_rom
                fZ = sp.interpolate.interp1d(time_pen[k0:k1], Z[:, k0:k1], kind='linear', fill_value='extrapolate')
                fMu = sp.interpolate.interp1d(time_pen[k0:k1], Mu[:, k0:k1], kind='linear', fill_value='extrapolate')
                a = (time_pen[k1-1] - time_pen[k0]) / 2
                b = (time_pen[k1-1] + time_pen[k0]) / 2
                time_k_lg = a * tlg + b
                Zk = fZ(time_k_lg)
                Muk = fMu(time_k_lg)
                for i in range(opt_obj.leggauss_deg):
                    grad_tensors[idx] += -a * wlg[i] * np.einsum('i,j', Muk[:, i], Zk[:, i])

        if opt_obj.which_fix == 'fix_bases':
            grad_Phi *= 0.0
            grad_Psi *= 0.0
            for k in range(len(grad_tensors)):
                grad_tensors[k] = sum(mpi_pool.comm.allgather(grad_tensors[k]))
        elif opt_obj.which_fix == 'fix_tensors':    
            for k in range(len(grad_tensors)):
                grad_tensors[k] *= 0.0
            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))
        else:
            grad_Phi = sum(mpi_pool.comm.allgather(grad_Phi))
            grad_Psi = sum(mpi_pool.comm.allgather(grad_Psi))
            for k in range(len(grad_tensors)):
                grad_tensors[k] = sum(mpi_pool.comm.allgather(grad_tensors[k]))

        if timing_enabled() and mpi_pool.rank == 0:
            print(
                "[timing] grad eval: "
                f"{tlib.perf_counter() - eval_t0:.3e}s total | "
                f"{format_runtime_profile()}"
            )
        return grad_Phi, grad_Psi, *grad_tensors
    
    return cost_poly, euclidean_gradient_poly, None


def check_gradient_using_finite_difference(M,Phi,Psi,A2,A3,A4,opt_obj,mpi_pool,fom,eps):

    cost, grad, _ = create_objective_and_gradient(M,opt_obj,mpi_pool,fom)
    gPhi, gPsi, gA2, gA3, gA4 = grad(Phi,Psi,A2,A3,A4)

    # Check Phi gradient 
    delta = sp.linalg.orth(np.random.randn(3,2))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi + eps*delta,Psi,A2,A3,A4) - cost(Phi - eps*delta,Psi,A2,A3,A4))
    dgrad = np.trace(delta.T@gPhi)
    error = np.abs(dfd - dgrad)
    percent_error = error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for Phi ------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check Psi gradient 
    delta = sp.linalg.orth(np.random.randn(3,2))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi + eps*delta,A2,A3,A4) - cost(Phi,Psi - eps*delta,A2,A3,A4))
    dgrad = np.trace(delta.T@gPsi)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for Psi ------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check A2 gradient 
    delta = np.random.randn(2,2)
    delta = delta/np.sqrt(np.trace(delta.T@delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi,A2 + eps*delta,A3,A4) - cost(Phi,Psi,A2 - eps*delta,A3,A4))
    dgrad = np.trace(delta.T@gA2)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A2 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")

    # Check A3 gradient 
    delta = np.random.randn(2,2,2)
    delta = delta/np.sqrt(np.einsum('ijk,ijk',delta,delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)

    dfd = (0.5/eps)*(cost(Phi,Psi,A2,A3 + eps*delta,A4) - cost(Phi,Psi,A2,A3 - eps*delta,A4))
    dgrad = np.einsum('ijk,ijk',delta,gA3)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A3 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")


    # Check A4 gradient 
    delta = np.random.randn(2,2,2,2)
    delta = delta/np.sqrt(np.einsum('ijkl,ijkl',delta,delta))
    if mpi_pool.rank == 0: 
        for k in range (1,mpi_pool.size):
            mpi_pool.comm.send(delta,dest=k)
    else:
        delta = mpi_pool.comm.recv(source=0)
        
    dfd = (0.5/eps)*(cost(Phi,Psi,A2,A3,A4 + eps*delta) - cost(Phi,Psi,A2,A3,A4 - eps*delta))
    dgrad = np.einsum('ijkl,ijkl',delta,gA4)
    error = np.abs(dfd - dgrad)
    percent_error = 100*error/np.abs(dfd)

    if mpi_pool.rank == 0:
        print("------ Error for A4 -------------")
        print("dfd = %1.5e,\t dfgrad = %1.5e,\t error = %1.5e,\t percent error = %1.5e"%(dfd,dgrad,error,percent_error))
        print("---------------------------------")
