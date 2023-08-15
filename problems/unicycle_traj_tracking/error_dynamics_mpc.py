import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType

"""
this ErrorDynamicsMPC class is used to solve tracking problem of uni-cycle model
using MPC. The error dynamics is defined as follows:
error dynamics:
    psi_dot = At * psi_t + Bt * ut + ht
state:
    psi: lie algebra element of Psi (SE2 error)
control:
    ut = xi_t: twist (se2 element)

State transition matrix:
    At: ad_{xi_d,t}
Control matrix:
    B_k = I
offset:
    ht = xi_t,d: desired twist (se2 element)
    
the reference trajectory is generated using TrajGenerator class in ref_traj_generator.py
"""


class ErrorDynamicsMPC:
    def __init__(self, ref_traj_config):
        self.nState = 3
        self.nControl = 3
        self.nTraj = None
        self.ref_traj = None
        self.ref_v = None
        self.dt = None
        self.Q = None
        self.R = None
        self.N = None
        self.setup_solver()
        self.set_ref_traj(ref_traj_config)
        self.setup_solver()

    def set_ref_traj(self, traj_config):
        traj_generator = TrajGenerator(traj_config)
        self.ref_traj, self.ref_v, self.dt = traj_generator.get_traj()
        self.nTraj = self.ref_traj.shape[1]

    def setup_solver(self, Q=10, R=1, N=5):
        self.Q = Q * np.diag(np.ones(self.nState))
        self.R = R * np.diag(np.ones(self.nControl))
        self.N = N

    def solve(self, state, t):
        """
        state: [x, y, theta] -> [x, y, cos(theta), sin(theta)]
        t: time -> index of reference trajectory (t = k * dt)
        """
        if self.ref_traj is None:
            raise ValueError('Reference trajectory is not set up yet!')

        # convert state to SE2 coeffs
        SE2_coeffs = state
        # get reference state and twist
        k = math.ceil(t / self.dt)
        ref_SE2_coeffs = self.ref_traj[:, k]

        # get error state and error dynamics
        psi_start = SE2(ref_SE2_coeffs).between(SE2(SE2_coeffs)).log().coeffs()
        Q = self.Q
        R = self.R
        N = self.N
        dt = self.dt


        # setup casadi solver
        opti = ca.Opti()
        psi_var = opti.variable(self.nState, N + 1)
        xi_var = opti.variable(2, N)

        # setup initial condition
        opti.subject_to(psi_var[:, 0] == psi_start)

        # setup dynamics constraints
        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            xi_goal = self.ref_v[:, index]  # desir
            A = -SE2Tangent(xi_goal).smallAdj()
            B = np.eye(self.nControl)
            h = -xi_goal
            psi_next = psi_var[:, i] + dt * (A @ psi_var[:, i] + B @ self._to_local_vel(xi_var[:, i]) + h)
            opti.subject_to(psi_var[:, i + 1] == psi_next)

        # cost function
        cost = 0
        for i in range(N):
            cost += ca.mtimes([psi_var[:, i].T, Q, psi_var[:, i]]) + ca.mtimes(
                [self._to_local_vel(xi_var[:, i]).T, R, self._to_local_vel(xi_var[:, i])])

        cost += ca.mtimes([psi_var[:, -1].T, 100 * Q, psi_var[:, -1]])
        opti.minimize(cost)
        opti.solver('ipopt')
        sol = opti.solve()
        psi_sol = sol.value(psi_var)
        xi_sol = sol.value(xi_var)

        return xi_sol[:, 0]

    def _to_local_vel(self, vel_cmd):
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])


def test_mpc():
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([1, 1, np.pi / 2]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 170,
                             'dt': 0.05}}

    mpc = ErrorDynamicsMPC(traj_config)
    ref_traj = mpc.ref_traj
    init_state = np.array([1, 1, 0])
    init_state = SE2Tangent(init_state).exp().coeffs()
    t = 0
    # contrainer to store state
    state_store = np.zeros((4, mpc.nTraj))
    state_store[:, 0] = init_state

    # start simulation
    for i in range(mpc.nTraj - 1):
        state = state_store[:, i]
        xi = mpc.solve(state, t)
        xi = mpc._to_local_vel(xi)
        X = SE2(state)  # SE2 state
        X = X + SE2Tangent(xi * mpc.dt)
        state_store[:, i + 1] = X.coeffs()
        t += mpc.dt

    # plot
    plt.figure()
    plt.plot(ref_traj[0, :], ref_traj[1, :], 'r')
    plt.plot(state_store[0, :], state_store[1, :], 'b')
    plt.legend(['reference', 'trajectory'])

    plt.show()

    # plot distance difference
    distance_store = np.linalg.norm(state_store[0:2, :] - ref_traj[0:2, :], axis=0)
    plt.figure()
    plt.plot(distance_store)
    plt.title('distance difference')
    plt.show()

    # plot orientation difference
    orientation_store = np.zeros(mpc.nTraj)
    for i in range(mpc.nTraj):
        X_d = SE2(ref_traj[:, i])
        X = SE2(state_store[:, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())

    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')

    plt.show()


if __name__ == '__main__':
    # test_generate_ref_traj()
    test_mpc()