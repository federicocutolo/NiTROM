import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import seaborn as sns
import scipy.integrate as integrate

sns.set_style("white")
sns.set_context("poster")

# Define H, dH/dp and -dH/dq
def Ham(p, q):
    return p**2/(2*m*l*l) + m*g*l*(1-np.cos(q))
def qdot(p):
    return p/(m*l*l)
def pdot(q):
    return -m*g*l*np.sin(q)

# Parameters for pendulum
m, l, g = 1.0, 1.0, 9.81

# Integration parameters
h = 0.01
t_end = 20*2*np.pi
N = int(t_end/h)
t = np.linspace(0, t_end, N+1)

# Initial conditions
q0 = np.pi/4
p0 = 0.0
z0 = q0, p0

##################################################
## EULER ##
##################################################
# Variables storage
p_e, p_se = np.zeros(N+1), np.zeros(N+1)
q_e, q_se = np.zeros(N+1), np.zeros(N+1)

p_e[0] = p0
p_se[0] = p0
q_e[0] = q0
q_se[0] = q0

for n in range(N):
    # Euler
    p_e[n+1] = p_e[n] + h*pdot(q_e[n])
    q_e[n+1] = q_e[n] + h*qdot(p_e[n])
    
    # Symplectic Euler: Attention! Conditionally stable!
    q_se[n+1] = q_se[n] + h*qdot(p_se[n])
    p_se[n+1] = p_se[n] + h*pdot(q_se[n+1])

H_e = Ham(p_e, q_e)
H_se = Ham(p_se, q_se)

##################################################
## RK4 ##
##################################################
q_rk4, p_rk4 = np.zeros(N+1), np.zeros(N+1)
q_rk4[0], p_rk4[0] = z0

# Use solve_ivp to integrate the system
def pendulum_ode(t, z):
    q, p = z
    return np.array([qdot(p), pdot(q)])

sol = integrate.solve_ivp(pendulum_ode, (0, t_end), z0, t_eval=t, method='RK45')

q_rk4 = sol.y[0]
p_rk4 = sol.y[1]
H_rk4 = Ham(p_rk4, q_rk4)
print(f"RK4 energy: {H_rk4-H_rk4[0]}")

##################################################
## Symplectic RK4: SRKN ##
##################################################
def SRKN_b6(z0, h):
    """Six-stage Symmetric Runge–Kutta–Nyström integrator.

    Parameters
    ----------
    z0 : tuple[float, float]
        Initial condition (q0, p0).
    h : float
        Time step size.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (q_rk, p_rk, H_rk) arrays of shape (N+1,) containing 
        variables, momenta, and Hamiltonian at each time step.
    """

    def _coefficients():
        a1 = 0.245298957184271
        a2 = 0.604872665711080
        a3 = 0.5 - (a1 + a2)
        a = np.flipud(np.array([a1, a2, a3, a3, a2, a1]))

        b1 = 0.0829844064174052
        b2 = 0.396309801498368
        b3 = -0.0390563049223486
        b4 = 1 - 2 * (b1 + b2 + b3)
        b = np.flipud(np.array([b1, b2, b3, b4, b3, b2, b1]))
        return a, b

    def _step(q, p, h, a, b):
        for idx in range(6):
            q = q + h * b[idx] * qdot(p)
            p = p + h * a[idx] * pdot(q)
        q = q + h * b[-1] * qdot(p)
        return q, p

    a, b = _coefficients()
    q_rk = np.zeros(N + 1)
    p_rk = np.zeros(N + 1)
    H_rk = np.zeros(N + 1)

    q_rk[0], p_rk[0] = z0
    H_rk[0] = Ham(p_rk[0], q_rk[0])

    current_state = z0
    for i in range(N):
        q_i, p_i = current_state
        q_i, p_i = _step(q_i, p_i, h, a, b)

        q_rk[i + 1], p_rk[i + 1] = q_i, p_i
        H_rk[i + 1] = Ham(p_i, q_i)
        current_state = (q_i, p_i)

    return q_rk, p_rk, H_rk

q_rk, p_rk, H_rk = SRKN_b6(z0, h)

# RK4


# Plots
plt.figure(figsize=(14, 7))
plt.tight_layout()
plt.suptitle("Symplectic schemes", fontsize=14)

plt.subplot(1,2,1)
# plt.plot(q_e, p_e, linestyle='dashed', label='Euler')
plt.plot(q_rk4, p_rk4, linestyle='dashed', label='RK4')
plt.plot(q_se, p_se, label='Symplectic Euler')
plt.plot(q_rk, p_rk, label='Symplectic RK4')
plt.title("Phase space")
plt.xlim([-1.0, 1.0])
plt.xlabel(r'$\theta$'); plt.ylabel('p')
plt.legend()

plt.subplot(1,2,2)
# plt.plot(t, np.abs(H_e-H_e[0]), linestyle='dashed', label='Euler')
plt.plot(t, np.abs(H_rk4-H_rk4[0]), linestyle='dashed', label='RK4')
plt.plot(t, np.abs(H_se-H_se[0]), label='Symplectic Euler')
plt.plot(t, np.abs(H_rk-H_rk[0]), label='Symplectic RK4')
plt.yscale('log')
plt.title("Energy")
plt.xlabel("t")
plt.ylabel("H-H0")
plt.legend()

# plt.subplot(1,3,3)
# plt.plot(t, q_e, linestyle='dashed', label='Euler')
# plt.plot(t, q_se, label='Symplectic Euler')
# plt.plot(t, q_rk, label='Symplectic RK4')
# plt.title("Angle (rad)")
# plt.xlabel("t")
# # plt.legend()
plt.savefig("./SymplecticSchemes_pendulum.pdf", bbox_inches='tight')
# plt.savefig("./SymplecticEuler_pendulum.pdf", bbox_inches='tight')


# SRKN convergence study
hs = np.array([0.1, 0.05, 0.025, 0.0125]) # h/2**i
h_ref = 0.00625
q_ref, p_ref, H_ref = SRKN_b6(z0, h_ref)
err_list = []
for hi in hs:
    q_rk, p_rk, H_rk = SRKN_b6(z0, hi)
    err_H = np.abs(H_rk[-1]-H_ref[-1])
    err_list.append(err_H)

log_err = np.log(err_list)
log_hs = np.log(hs)
slope, intercept = np.polyfit(log_hs, log_err, 1)
x = np.linspace(np.max(log_hs), np.min(log_hs), len(hs))
y = slope*x + intercept
print(f"Convergence order: {slope:.4f}")

# Convergence plot
plt.figure()
plt.loglog(hs, np.array(err_list), marker='o', label="SRKN_b6")
plt.loglog(hs, np.exp(y), linestyle='dashed', label=f"slope={slope:.2f}")
plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("err_H")
plt.legend()
plt.title("Method convergence")
plt.savefig("./SRKN_b6_convergence.pdf", bbox_inches='tight')









plt.show()