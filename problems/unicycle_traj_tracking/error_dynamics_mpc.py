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
        self.nState = 3  # twist error (se2 vee) R^3
        self.nControl = 2  # velocity control (v, w) R^2
        self.nTwist = 3  # twist (se2 vee) R^3
        self.nTraj = None
        self.ref_SE2 = None
        self.ref_twist = None
        self.dt = None
        self.Q = None
        self.R = None
        self.N = None
        self.setup_solver()
        self.set_ref_traj(ref_traj_config)
        self.setup_solver()

    def set_ref_traj(self, traj_config):
        traj_generator = TrajGenerator(traj_config)
        self.ref_SE2, self.ref_twist, self.dt = traj_generator.get_traj()
        self.nTraj = self.ref_SE2.shape[1]

    def setup_solver(self, Q=10, R=1, N=10):
        self.Q = Q * np.diag(np.ones(self.nState))
        self.R = R * np.diag(np.ones(self.nControl))
        self.N = N

    def solve(self, SE2_coeffs, t):
        """
        SE2_coeffs: [x, y, cos(theta), sin(theta)] (SE2 coeffs of current state)
        t: time -> index of reference trajectory (t = k * dt)

        return:
            u: control input (v, w)
        """
        if self.ref_SE2 is None:
            raise ValueError('Reference trajectory is not set up yet!')

        # get reference state and twist
        k = math.ceil(t / self.dt)
        curr_ref_SE2_coeffs = self.ref_SE2[:, k]

        # get x init by calculating log between current state and reference state
        x_init = SE2(curr_ref_SE2_coeffs).between(SE2(SE2_coeffs)).log().coeffs()
        Q = self.Q
        R = self.R
        N = self.N
        dt = self.dt


        # setup casadi solver
        opti = ca.Opti()
        x_var = opti.variable(self.nState, N + 1)
        u_var = opti.variable(2, N)

        # setup initial condition
        opti.subject_to(x_var[:, 0] == x_init)

        # setup dynamics constraints
        # x_next = A * x + B * u + h
        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            u_d = self.ref_twist[:, index]  # desir
            A = -SE2Tangent(u_d).smallAdj()
            B = np.eye(self.nTwist)
            h = -u_d
            x_next = x_var[:, i] + dt * (A @ x_var[:, i] + B @ self._to_local_twist(u_var[:, i]) + h)
            opti.subject_to(x_var[:, i + 1] == x_next)

        # cost function
        cost = 0
        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            twist_ref = self.ref_twist[:, index]
            u_d = self._to_vel_cmd(twist_ref)
            cost += ca.mtimes([x_var[:, i].T, Q, x_var[:, i]]) + ca.mtimes(
                [(u_var[:, i]-u_d).T, R, (u_var[:, i]-u_d)])

        cost += ca.mtimes([x_var[:, -1].T, 100 * Q, x_var[:, -1]])
        opti.minimize(cost)
        opti.solver('ipopt')
        sol = opti.solve()
        psi_sol = sol.value(x_var)
        u_sol = sol.value(u_var)

        return u_sol[:, 0]

    def _to_local_twist(self, vel_cmd):
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def _to_vel_cmd(self, local_vel):
        return ca.vertcat(local_vel[0], local_vel[2])


def test_mpc():
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([1, 1, np.pi / 2]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 170,
                             'dt': 0.05}}

    mpc = ErrorDynamicsMPC(traj_config)
    ref_SE2 = mpc.ref_SE2
    init_state = np.array([0.5, 0.5, 0])
    init_state = SE2Tangent(init_state).exp().coeffs()
    t = 0
    # contrainer to store state
    state_store = np.zeros((4, mpc.nTraj))
    state_store[:, 0] = init_state

    # start simulation
    for i in range(mpc.nTraj - 1):
        state = state_store[:, i]
        vel_cmd = mpc.solve(state, t)
        xi = mpc._to_local_twist(vel_cmd)
        X = SE2(state)  # SE2 state
        X = X + SE2Tangent(xi * mpc.dt)
        state_store[:, i + 1] = X.coeffs()
        t += mpc.dt

    # plot
    plt.figure()
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    plt.plot(state_store[0, :], state_store[1, :], 'b')
    plt.legend(['reference', 'trajectory'])

    plt.show()

    # plot distance difference
    distance_store = np.linalg.norm(state_store[0:2, :] - ref_SE2[0:2, :], axis=0)
    plt.figure()
    plt.plot(distance_store)
    plt.title('distance difference')
    plt.show()

    # plot orientation difference
    orientation_store = np.zeros(mpc.nTraj)
    for i in range(mpc.nTraj):
        X_d = SE2(ref_SE2[:, i])
        X = SE2(state_store[:, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())

    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')

    plt.show()

def test_pose_regulation():
    init_state = np.array([5, 5, 0])
    config = {'type': TrajType.POSE_REGULATION,
              'param': {'end_state': np.array([0, 0,- np.pi / 2]),
                        'dt': 0.05,
                        'nTraj': 670}}
    traj_generator = TrajGenerator(config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    mpc = ErrorDynamicsMPC(config)
    # container for recording SE2 state and twist
    nSE2 = 4
    nTwist = 3
    store_SE2 = np.zeros((nSE2, ref_SE2.shape[1]))
    store_twist = np.zeros((nTwist, ref_SE2.shape[1]))
    store_SE2[:, 0] = SE2(init_state[0], init_state[1], init_state[2]).coeffs()

    t = 0
    for i in range(ref_SE2.shape[1] - 1):
        curr_SE2 = store_SE2[:, i]
        curr_ref_SE2 = ref_SE2[:, i]
        curr_ref_twist = ref_twist[:, i]
        curr_vel_cmd = mpc.solve(curr_SE2,t)
        curr_twist = mpc._to_local_twist(curr_vel_cmd)
        store_twist[:, i] = curr_twist.full().flatten()
        # next SE2 state
        next_SE2 = SE2(curr_SE2) + SE2Tangent(curr_twist) * dt
        store_SE2[:, i + 1] = next_SE2.coeffs()
        t += dt

    # plot
    plt.figure()
    plt.plot(store_SE2[0, :], store_SE2[1, :], 'b')
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    plt.show()

    # # plot (x, y, theta) pose figure with arrow
    # plt.figure()
    # plt.plot(store_SE2[0, :], store_SE2[1, :], 'b')
    # plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    # plt.quiver(store_SE2[0, :], store_SE2[1, :], np.cos(store_SE2[2, :]), np.sin(store_SE2[2, :]), color='b')
    # plt.quiver(ref_SE2[0, :], ref_SE2[1, :], np.cos(ref_SE2[2, :]), np.sin(ref_SE2[2, :]), color='r')
    # plt.show()

    # plot distance error
    plt.figure()
    plt.plot(np.linalg.norm(store_SE2[0:2, :] - ref_SE2[0:2, :], axis=0))
    plt.title('distance error')
    plt.show()

    # plot angle error
    plt.figure()
    orientation_store = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        X_d = SE2(ref_SE2[:, i])
        X = SE2(store_SE2[:, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())

    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')
    plt.show()

if __name__ == '__main__':
    # test_generate_ref_traj()
    test_pose_regulation()
