import numpy as np 
import scipy 
import matplotlib.pyplot as plt
from mpi4py import MPI

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sys
import os

# Ensure the top-level repository root is on sys.path so NiTROM package can be imported.
HERE = os.path.dirname(os.path.abspath(__file__))        # <repo>/NiTROM/Examples/toymodel
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))  # <repo>/NiTROM
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pymanopt
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from pymanopt.tools.diagnostics import check_gradient

plt.rcParams.update({"font.family":"serif","font.sans-serif":["Computer Modern"],'font.size':18,'text.usetex':True})
plt.rc('text.latex',preamble=r'\usepackage{amsmath}')

from NiTROM.PyManopt_Functions.my_pymanopt_classes import myAdaptiveLineSearcher
from NiTROM.Optimization_Functions import (classes, 
                                           nitrom_functions, 
                                           opinf_functions as opinf_fun, 
                                           troop_functions)
from NiTROM.Optimization_Functions import opinf_functions_energypreserving as opinf_fun_ep
import fom_class


cPOD, cOI, cOIEP, cTR, cOPT = '#66c2a5', '#fc8d62', '#fc8d62', '#8da0cb', '#e78ac3'
lPOD, lOI, lOIEP, lTR, lOPT = 'solid', 'dotted', 'solid', 'dashed', 'dashdot'


#%% Instantiate the full-order model class

n = 3 
beta = 20
A2 = np.diag([-1,-2,-5])
A3 = np.zeros((3,3,3))
A3[:,:,-1] = np.diag([beta,beta,0.0])
B = np.ones((3,1))
C = np.ones((1,3))
time = np.linspace(0,10,num=100)

fom = fom_class.full_order_model(A2,A3,B,C)

#%% Generate training trajectories and save to file

traj_path = os.path.join(HERE, "trajectories/") 
os.makedirs(traj_path, exist_ok=True)

fname_traj = traj_path + "traj_%03d.npy"
fname_weight = traj_path + "weight_%03d.npy"
fname_forcing = traj_path + "forcing_%03d.npy"
fname_deriv = traj_path + "deriv_%03d.npy"
fname_time = traj_path + "time.npy"

n_traj = 4

max_val = 5/20
# max_val = 1.0

betas = np.asarray([0.01,0.1,0.2,0.248])
# betas = max_val*np.asarray([0.05,0.4,0.8,0.99])
n_traj = len(betas)
weights = np.zeros(len(betas))
for k in range (len(betas)):
    u = betas[k]*np.ones(3)
    
    sol = solve_ivp(fom.evaluate_fom_dynamics,
                    [0,time[-1]],np.zeros(3),
                    'RK45',
                    t_eval=time,
                    args=(u,))
    
    dX = np.zeros((3,len(time)))
    for j in range (sol.y.shape[-1]):
        dX[:,j] = fom.evaluate_fom_dynamics(time[j],sol.y[:,j],u) - u
    
    id_ss = np.asarray([-betas[k]/(-1+beta/5*betas[k]),-betas[k]/(-2+beta/5*betas[k]),betas[k]/5])
    weights[k] = np.linalg.norm(fom.compute_output(id_ss))**2
    
    np.save(fname_traj%k,sol.y)
    np.save(fname_deriv%k,dX)
    np.save(fname_weight%k,[weights[k]])
    np.save(fname_forcing%k,u)
    

np.save(traj_path + "time.npy",time)


#%% Compute POD model 

pool_inputs = (MPI.COMM_WORLD, n_traj, fname_traj, fname_time)
pool_kwargs = {'fname_steady_forcing':fname_forcing,'fname_weights':fname_weight,'fname_derivs':fname_deriv}
pool = classes.mpi_pool(*pool_inputs,**pool_kwargs)


r = 2               # ROM dimension
poly_comp = [1,2]   # Model with a linear part and a quadratic part


Phi_pod, _ = opinf_fun.perform_POD(pool,2)
Psi_pod = Phi_pod.copy()
tensors_pod, _ = fom.assemble_petrov_galerkin_tensors(Phi_pod,Psi_pod)


#%% Compute NiTROM model 

which_trajs = np.arange(0,pool.my_n_traj,1)
which_times = np.arange(0,pool.n_snapshots,1)
leggauss_deg = 5
nsave_rom = 2

opt_obj_inputs = (pool,
                  which_trajs,
                  which_times,
                  leggauss_deg,
                  nsave_rom,
                  [1,2])
# opt_obj_kwargs = {'stab_promoting_pen':1e-2,'stab_promoting_tf':20,'stab_promoting_ic':(np.random.randn(r),)}


opt_obj = classes.optimization_objects(*opt_obj_inputs) #**opt_obj_kwargs)


St = manifolds.Stiefel(n,r)
Gr = manifolds.Grassmann(n,r)
Euc_rr = manifolds.Euclidean(r,r)
Euc_rrr = manifolds.Euclidean(r,r,r)

M = manifolds.Product([Gr,St,Euc_rr,Euc_rrr])
cost, grad, hess = nitrom_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
problem = pymanopt.Problem(M,cost,euclidean_gradient=grad)
# check_gradient(problem,x=[Phi_pod,Psi_pod,*tensors_pod])
# check_gradient(problem,x=result.point)


line_searcher = myAdaptiveLineSearcher(contraction_factor=0.5,sufficient_decrease=0.85,max_iterations=25,initial_step_size=1)
optimizer = optimizers.ConjugateGradient(max_iterations=5,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1)


point = (Phi_pod,Psi_pod) + tensors_pod
print(cost(*point))
grad_val = grad(*point)
norm = 0.0
for tensor in grad_val:
    norm += np.linalg.norm(tensor)
print(norm)
result = optimizer.run(problem,initial_point=point)
# check_gradient(problem,x=result.point)

Phi_nit = result.point[0]
Psi_nit = result.point[1]
Phi_nit = Phi_nit@scipy.linalg.inv(Psi_nit.T@Phi_nit)
tensors_nit = tuple(result.point[2:])


itervec_nit = result.log["iterations"]["iteration"]
costvec_nit = result.log["iterations"]["cost"]
gradvec_nit = result.log["iterations"]["gradient_norm"]

#%% Compute TrOOP model

optimizer = optimizers.ConjugateGradient(max_iterations=5,min_step_size=1e-20,max_time=3600,line_searcher=line_searcher,log_verbosity=1,min_gradient_norm=1e-7)

M = manifolds.Product([Gr,Gr])
cost_troop, grad_troop, _ = troop_functions.create_objective_and_gradient(M,opt_obj,pool,fom)
problem = pymanopt.Problem(M,cost_troop,euclidean_gradient=grad_troop)


result = optimizer.run(problem,initial_point=(Phi_pod,Psi_pod))
# check_gradient(problem,x=result.point)

Phi_tr = result.point[0]
Psi_tr = result.point[1]
Phi_tr = Phi_tr@scipy.linalg.inv(Psi_tr.T@Phi_tr)

tensors_tr, _ = fom.assemble_petrov_galerkin_tensors(Phi_tr,Psi_tr)

itervec_tr = result.log["iterations"]["iteration"]
costvec_tr = result.log["iterations"]["cost"]
gradvec_tr = result.log["iterations"]["gradient_norm"]


#%%
plt.figure()
plt.plot(itervec_tr,costvec_tr,color=cTR,linestyle=lTR,label='TrOOP')
plt.plot(itervec_nit,costvec_nit,color=cOPT,linestyle=lOPT,label='NiTROM')

ax = plt.gca()
ax.set_yscale('log')
ax.set_xlabel('Conj. gradient iteration')
ax.set_ylabel('Cost')

# ax.set_box_aspect(0.30)

plt.legend()
plt.tight_layout()

# plt.savefig("Figures/convergence_plot_beta%d.eps"%beta,format='eps')


#%% Compute OpInf model

# weights = pool.weights.copy()
# pool.weights *= pool.n_traj*pool.n_snapshots

lam = np.logspace(-8,-2,num=1)
cost_oi = []
for (count,l) in enumerate(lam):
    tensors_opinf = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,[0.0,l])
    point = (Phi_pod,Psi_pod) + tensors_opinf
    cost_oi.append(cost(*point))

# pool.weights = weights

#%%
plt.figure()
plt.plot(lam,cost_oi)

#%%
lambdas = [0.0,lam[np.argmin(cost_oi)]]
print(np.min(cost_oi),lambdas)
tensors_oi = opinf_fun.operator_inference(pool,Phi_pod,poly_comp,lambdas)


lam = np.logspace(-8,-2,num=1)
cost_oi = []
for (count,l) in enumerate(lam):
    tensors_opinf_ep = opinf_fun_ep.opinf_ep(pool,Phi_pod,[0.0,l])
    point = (Phi_pod,Psi_pod) + tensors_opinf_ep
    cost_oi.append(cost(*point))

lambdas = [0.0,lam[np.argmin(cost_oi)]]
print(np.min(cost_oi),lambdas)
tensors_oi_ep = opinf_fun_ep.opinf_ep(pool,Phi_pod,lambdas)

#%%

betas = np.random.uniform(0.0,0.999*max_val,size=100)
# betas = [0.2]
t_eval = np.linspace(0,10,num=100)

error_pod = np.zeros_like(t_eval)
error_tr = np.zeros_like(t_eval)
error_oi = np.zeros_like(t_eval)
error_oi_ep = np.zeros_like(t_eval)
error_nit = np.zeros_like(t_eval)

for k in range (len(betas)):
    u = betas[k]*np.ones(3)
    
    
    sol = solve_ivp(fom.evaluate_fom_dynamics,[0,time[-1]],np.zeros(3),'RK45',t_eval=t_eval,args=(u,)).y
    id_ss = np.asarray([-betas[k]/(-1+4*betas[k]),-betas[k]/(-2+4*betas[k]),betas[k]/5])
    weight = np.linalg.norm(fom.compute_output(id_ss))**2
    
    
    sol_pod = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(Phi_pod.T@u,) + tensors_pod)).y
    sol_tr = Phi_tr@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(Psi_tr.T@u,) + tensors_tr)).y
    sol_nit = Phi_nit@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(Psi_nit.T@u,) + tensors_nit)).y
    sol_oi = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(Phi_pod.T@u,) + tensors_oi)).y
    sol_oi_ep = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,time[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(Phi_pod.T@u,) + tensors_oi_ep)).y
    error_pod += np.linalg.norm(C@(sol_pod - sol),axis=0)**2/weight/len(betas)
    error_tr += np.linalg.norm(C@(sol_tr - sol),axis=0)**2/weight/len(betas)
    error_oi += np.linalg.norm(C@(sol_oi - sol),axis=0)**2/weight/len(betas)
    error_oi_ep += np.linalg.norm(C@(sol_oi_ep - sol),axis=0)**2/weight/len(betas)
    error_nit += np.linalg.norm(C@(sol_nit - sol),axis=0)**2/weight/len(betas)
    
plt.figure()
plt.plot(t_eval,error_pod,color=cPOD,linestyle=lPOD,label='POD Gal.')
plt.plot(t_eval,error_oi,color=cOI,linestyle=lOI,label='OpInf')
plt.plot(t_eval,error_oi_ep,color=cOIEP,linestyle=lOIEP,label='OpInf_EP')
plt.plot(t_eval,error_tr,color=cTR,linestyle=lTR,label='TrOOP')
plt.plot(t_eval,error_nit,color=cOPT,linestyle=lOPT,label='NiTROM')


ax = plt.gca()
ax.set_xlabel('Time $t$')
ax.set_ylabel('Average error $e(t)$')
ax.set_yscale('log')

plt.legend()
plt.tight_layout()

plt.savefig("Figures/testing_error_beta%d.png"%beta, dpi=300)
    

#%%
tk = np.linspace(0,30,num=10000)
uk = 0.45*(np.sin(tk) + np.cos(2*tk))

t_eval = np.linspace(0,30,num=1000)


fu = scipy.interpolate.interp1d(tk,np.outer(B,uk),kind='linear',fill_value='extrapolate')
sol_fom = (solve_ivp(fom.evaluate_fom_dynamics,[0,tk[-1]],np.zeros(3),'RK45',t_eval=t_eval,args=(fu,))).y



fu = scipy.interpolate.interp1d(tk,np.outer(Psi_pod.T@B,uk),kind='linear',fill_value='extrapolate')
sol_pod = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,tk[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(fu,) + tensors_pod)).y


fu = scipy.interpolate.interp1d(tk,np.outer(Psi_pod.T@B,uk),kind='linear',fill_value='extrapolate')
sol_oi = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,tk[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(fu,) + tensors_oi)).y
sol_oi_ep = Phi_pod@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,tk[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(fu,) + tensors_oi_ep)).y


fu = scipy.interpolate.interp1d(tk,np.outer(Psi_tr.T@B,uk),kind='linear',fill_value='extrapolate')
sol_tr = Phi_tr@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,tk[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(fu,) + tensors_tr)).y


fu = scipy.interpolate.interp1d(tk,np.outer(Psi_nit.T@B,uk),kind='linear',fill_value='extrapolate')
sol_nit = Phi_nit@(solve_ivp(opt_obj.evaluate_rom_rhs,[0,tk[-1]],np.zeros(r),'RK45',t_eval=t_eval,args=(fu,) + tensors_nit)).y


plt.figure()
plt.plot(t_eval,fom.compute_output(sol_fom)[0,],color='k',linewidth=2,label='FOM')
plt.plot(t_eval,fom.compute_output(sol_pod)[0,],color=cPOD,linestyle=lPOD,linewidth=2,label='POD Gal.')
# plt.plot(t_eval,fom.compute_output(sol_oi)[0,],color=cOI,linestyle=lOI,linewidth=2)
plt.plot(t_eval,fom.compute_output(sol_oi_ep)[0,],color=cOIEP,linestyle=lOIEP,linewidth=2,label='OpInf_EP')
plt.plot(t_eval,fom.compute_output(sol_tr)[0,],color=cTR,linestyle=lTR,linewidth=2,label='TrOOP')
plt.plot(t_eval,fom.compute_output(sol_nit)[0,],color=cOPT,linestyle=lOPT,linewidth=2,label='NiTROM')

ax = plt.gca()
ax.set_xlabel('Time $t$')
ax.set_ylabel('$y(t)$')
ax.set_title('$u(t) = 0.45(\sin(t) + \cos(2t))$')

plt.legend()
plt.tight_layout()

plt.savefig("Figures/sinusoid_beta%d.png"%beta, dpi=300)
    
    
    
    
    
    
    
    
    
    
    
    



