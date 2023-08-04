"""
se2 tracking problem

error dynamics:
    x_k+1 = A_k * x_k + B_k * u_k
state:
    x = [psi, 1]
    psi: lie algebra element of Psi (SE2 error)
control:
    u = xi: twist (se2 element)
State transition matrix:
    A_k = [I+dt*At dt*ht; 0 1]
    At: ad_{xi_d,t}
    ht: xi_d,t
Control matrix:
    B_k = [dt*I; 0]

"""
import numpy as np
import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
from pyMPC.mpc import MPCController
from scipy.integrate import ode
from manifpy import SE2, SE2Tangent
import casadi as ca
def ad_se2(xi):
    """
    Adjoint map of se2
    """
    xi1, xi2, xi3 = xi
    return np.array([[0, -xi3, -xi2],
                     [xi3, 0, xi1],
                     [0, 0, 0]])


def solver():
    nx, nu = 3, 3
    dt = 0.05
    N = 100
    Q = 10*np.diag([1, 1, 1])
    R = np.diag([1, 1, 1])
    xi_goal = np.array([1, 0, 0])
    A = -SE2Tangent(xi_goal).smallAdj()
    B = np.eye(nu)
    h = -xi_goal
    # psi_start = np.array([1, 1, 1])
    psi_start = np.array([0, 0, 0])

    # define opti solver
    opti = ca.Opti()
    psi_var = opti.variable(nx, N+1)  # error state variable
    u_var = opti.variable(nu, N)      # control variable

    # initial error condition
    opti.subject_to(psi_var[:, 0] == psi_start)

    # system model constraints
    for i in range(N):
        psi_next = psi_var[:, i] + dt * (A @ psi_var[:, i] + B @ u_var[:, i] + h)
        opti.subject_to(psi_var[:, i+1] == psi_next)

    # cost function
    cost = 0
    for i in range(N):
        cost += ca.mtimes([psi_var[:, i].T, Q, psi_var[:, i]]) + ca.mtimes([u_var[:, i].T, R, u_var[:, i]])

    cost += ca.mtimes([psi_var[:, -1].T, 10000*Q, psi_var[:, -1]])
    opti.minimize(cost)
    opti.solver('ipopt')
    sol = opti.solve()
    psi_sol = sol.value(psi_var)
    u_sol = sol.value(u_var)
    return psi_sol, u_sol

if __name__ == '__main__':
    psi_sol, u_sol = solver()
    print(psi_sol)
    print(u_sol)
    plt.plot(psi_sol.T)
    legend = ['$\psi_1 = e_x$', '$\psi_2 = e_y$', '$\psi_3= e_{theta}$']
    plt.legend(legend)
    plt.show()

    plt.plot(u_sol.T)
    plt.legend(['$u_1 = v_x$', '$u_2 = v_y$', '$u_3= v_{theta}$'])
    plt.show()
