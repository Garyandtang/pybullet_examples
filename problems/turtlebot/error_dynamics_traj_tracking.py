import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math

"""
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
"""


class ErrorDynamicsMPC():
    def __init__(self, ref_traj_dict):
        self.nState = 3
        self.nControl = 3
        self.nTraj = 400
        self.dt = 0.05
        self.setup_solver()
        self.setup_ref_traj(ref_traj_dict)

    def setup_ref_traj(self, ref_traj_config):
        if not ref_traj_config:
            ref_traj_config = {'start_state': np.array([0, 0, 0]),
                               'linear_vel': 0.5,
                               'angular_vel': 0.5}
        start_state = ref_traj_config['start_state']
        ref_vel = np.array([ref_traj_config['linear_vel'], ref_traj_config['angular_vel']])
        xi = self.vel_cmd_to_local_vel(ref_vel, start_state)
        state = start_state
        # write a contrainer to store the reference trajectory
        self.ref_traj = np.zeros((4, self.nTraj))  # [x, y, cos(theta), sin(theta)]
        self.ref_v = np.zeros((3, self.nTraj))
        self.ref_traj[:, 0] = SE2(state[0], state[1], state[2]).coeffs()
        self.ref_v[:, 0] = xi

        for i in range(self.nTraj - 1):
            SE2_coeffs = self.ref_traj[:, i]
            X = SE2(SE2_coeffs)  # SE2 state
            X = X + SE2Tangent(xi * self.dt)  # X * SE2Tangent(xi * self.dt).exp()
            self.ref_traj[:, i + 1] = X.coeffs()
            self.ref_v[:, i + 1] = xi
        return self.ref_traj, self.ref_v, self.dt

    def setup_ref_traj_naive(self, ref_traj_config):
        if not ref_traj_config:
            ref_traj_config = {'start_state': np.array([0, 0, 0]),
                               'linear_vel': 1,
                               'angular_vel': 0.4}
        psi_start = ref_traj_config['start_state']
        ref_vel = np.array([ref_traj_config['linear_vel'], ref_traj_config['angular_vel']])
        psi = psi_start
        # write a contrainer to store the reference trajectory
        self.ref_traj = np.zeros((self.nState, self.nTraj))
        self.ref_v = np.zeros((3, self.nTraj))
        self.ref_traj[:, 0] = psi_start
        xi = self.vel_cmd_to_local_vel(ref_vel, psi_start)
        self.ref_v[:, 0] = xi
        for i in range(self.nTraj - 1):
            psi = self.ref_traj[:, i]
            psi = psi + self.vel_cmd_to_local_vel(ref_vel, psi) * self.dt
            self.ref_traj[:, i + 1] = psi
            self.ref_v[:, i + 1] = xi
        return self.ref_traj, self.ref_v, self.dt

    def setup_solver(self):
        self.Q = 10 * np.diag(np.ones(self.nState))
        self.R = np.diag(np.ones(self.nControl))
        self.N = 5

    def p_control_forward(self, state, t):
        """
        state: [x, y, theta] -> [x, y, cos(theta), sin(theta)]
        t: time -> index of reference trajectory (t = k * dt)
        """
        # convert state to SE2 coeffs
        SE2_coeffs = state
        # get reference state and twist
        k = math.ceil(t / self.dt)
        ref_SE2_coeffs = self.ref_traj[:, k]
        xi_goal = self.ref_v[:, k]  # desired twist
        X_ref = SE2(ref_SE2_coeffs)
        X = SE2(SE2_coeffs)
        trans_error = X.translation() - X_ref.translation()
        position_dff = scipy.linalg.norm(X_ref.translation() - X.translation())
        if trans_error[1] < 0:
            position_dff = - position_dff
        orientation_dff = 1 - np.cos(X_ref.angle() - X.angle())
        vel_cmd = 1*np.array([position_dff, orientation_dff])
        return np.array([vel_cmd[0], 0, vel_cmd[1]]) + xi_goal

    def p_control(self, state, t):
        """
        state: [x, y, theta] -> [x, y, cos(theta), sin(theta)]
        t: time -> index of reference trajectory (t = k * dt)
        """
        # convert state to SE2 coeffs
        SE2_coeffs = state
        # get reference state and twist
        k = math.ceil(t / self.dt)
        ref_SE2_coeffs = self.ref_traj[:, k]
        xi_goal = self.ref_v[:, k]  # desired twist
        X_ref = SE2(ref_SE2_coeffs)
        X = SE2(SE2_coeffs)
        X_diff = X.between(X_ref)
        xx = X_diff.log().coeffs()
        xi_feedback = np.array([0.7, 0, 1]) * self._to_local_vel(self.local_vel_to_vel_cmd(xx))
        # xi_feedback = np.array([3, 1, 1]) * self._to_local_vel(np.array([np.sqrt(xx[0] ** 2 + xx[1] ** 2), xx[2]]))
        xi = xi_feedback + xi_goal
        return xi

    def p_control_1(self, state, desired_state):
        """
        state: [x, y, theta] -> [x, y, cos(theta), sin(theta)]
        t: time -> index of reference trajectory (t = k * dt)
        """
        # convert state to SE2 coeffs
        SE2_coeffs = state
        # get reference state and twist
        X_ref = SE2(desired_state)
        X = SE2(SE2_coeffs)
        X_diff = X.between(X_ref)
        xx = X_diff.log().coeffs()
        # xi_feedback = np.array([1, 1, 1]) * self._to_local_vel(self.local_vel_to_vel_cmd(xx))
        xi_feedback = np.array([1, 1, 1]) * self._to_local_vel(np.array([np.sqrt(xx[0] ** 2 + xx[1] ** 2), xx[2]]))
        xi = xi_feedback
        return xx

    def solve(self, state, t):
        """
        state: [x, y, theta] -> [x, y, cos(theta), sin(theta)]
        t: time -> index of reference trajectory (t = k * dt)
        """
        # convert state to SE2 coeffs
        SE2_coeffs = state
        # get reference state and twist
        k = math.ceil(t / self.dt)
        ref_SE2_coeffs = self.ref_traj[:, k]
        xi_goal = self.ref_v[:, k]  # desired twist

        # get error state and error dynamics
        psi_start = SE2(ref_SE2_coeffs).between(SE2(SE2_coeffs)).log().coeffs()
        Q = self.Q
        R = self.R
        N = self.N
        dt = self.dt
        A = -SE2Tangent(xi_goal).smallAdj()
        B = np.eye(self.nControl)
        h = -xi_goal

        # setup casadi solver
        opti = ca.Opti()
        psi_var = opti.variable(self.nState, N + 1)
        xi_var = opti.variable(2, N)

        # setup initial condition
        opti.subject_to(psi_var[:, 0] == psi_start)

        # setup dynamics constraints
        for i in range(N):
            psi_next = psi_var[:, i] + dt * (A @ psi_var[:, i] + B @ self._to_local_vel(xi_var[:, i]) + h)
            opti.subject_to(psi_var[:, i + 1] == psi_next)

        # cost function
        cost = 0
        for i in range(N):
            cost += ca.mtimes([psi_var[:, i].T, Q, psi_var[:, i]]) + ca.mtimes([self._to_local_vel(xi_var[:, i]).T, R, self._to_local_vel(xi_var[:, i])])

        cost += ca.mtimes([psi_var[:, -1].T, 100 * Q, psi_var[:, -1]])
        opti.minimize(cost)
        opti.solver('ipopt')
        sol = opti.solve()
        psi_sol = sol.value(psi_var)
        xi_sol = sol.value(xi_var)

        return self._to_local_vel(xi_sol[:, 0])

    def _to_vel_cmd(self):
        pass

    def _to_local_vel(self, vel_cmd):
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def vel_cmd_to_local_vel(self, vel_cmd, psi):
        """
        Convert velocity command [linear_v, angular_v] to  velocity in local frame of robot
        :return:
        """

        return np.array([vel_cmd[0], 0, vel_cmd[1]])


    def local_vel_to_vel_cmd(self, local_vel):
        """
        Convert velocity in local frame of robot to velocity command [linear_v, angular_v]
        :return:
        """
        return np.array([0.5*local_vel[0] + 0.5*local_vel[1], local_vel[2]])
        # return np.array([np.sqrt(local_vel[0] ** 2 + local_vel[1] ** 2), local_vel[2]])

    def get_init_psi(self):
        pass


def test_generate_ref_traj():
    ref_traj_config = {'start_state': np.array([0, 0, 0]),
                       'linear_vel': 0.5,
                       'angular_vel': 0.5}
    mpc = ErrorDynamicsMPC(ref_traj_config)

    ref_traj, ref_v, dt = mpc.setup_ref_traj(ref_traj_config)
    ref_traj_naive, ref_v_naive, dt_naive = mpc.setup_ref_traj_naive(ref_traj_config)
    plt.figure()
    plt.plot(ref_traj[0, :], ref_traj[1, :])
    plt.show()

    # plot reference traj
    plt.figure()
    plt.plot(ref_traj[0, :], ref_traj[1, :])
    plt.show()

    # plot reference trajectories
    plt.figure()
    plt.plot(ref_traj[0, :])
    plt.plot(ref_traj[1, :])
    ref_theta = np.arctan2(ref_traj[3, :], ref_traj[2, :])
    plt.plot(ref_theta)
    plt.legend(['x', 'y', 'theta'])
    plt.title('reference trajectory')
    plt.show()

    # plot naive reference trajectories
    plt.figure()
    plt.plot(ref_traj_naive[0, :])
    plt.plot(ref_traj_naive[1, :])
    plt.plot(ref_traj_naive[2, :])
    plt.legend(['x', 'y', 'theta'])
    plt.title('naive reference trajectory')
    plt.show()


def test_solve():
    ref_traj_config = {'start_state': np.array([0, 0, 0]),
                       'linear_vel': 0.5,
                       'angular_vel': 0.5}
    mpc = ErrorDynamicsMPC(ref_traj_config)
    mpc.setup_solver()
    state = np.array([0, 0, 0])
    t = 0
    xi = mpc.solve(state, t)
    print(xi)


def test_pid():
    ref_traj_config = {'start_state': np.array([1, 1, 0]),
                       'linear_vel': 0.5,
                       'angular_vel': 0.5}
    mpc = ErrorDynamicsMPC(ref_traj_config)
    ref_traj, ref_v, dt = mpc.setup_ref_traj(ref_traj_config)
    state = np.array([-0.1, -0.1, 0])
    state = SE2(state[0], state[1], state[2]).coeffs()
    pid_res_store = np.zeros((4, mpc.nTraj))
    pid_res_store[:, 0] = state
    # pid control
    t = 0
    desired_state = np.array([-0.12, -0.1, np.pi/2])
    desired_state = SE2(desired_state[0], desired_state[1], desired_state[2]).coeffs()
    for i in range(mpc.nTraj - 1):
        state = pid_res_store[:, i]
        xi = mpc.p_control_1(state, desired_state)
        # print(xi)
        X = SE2(state)
        print(X)
        X = X + SE2Tangent(xi * mpc.dt)
        pid_res_store[:, i + 1] = X.coeffs()
        t += mpc.dt
    # plot state
    plt.figure()
    # plt.plot(ref_traj[0, :], ref_traj[1, :])
    plt.plot(pid_res_store[0, :], pid_res_store[1, :])
    # plt.legend(['ref', 'pid'])
    plt.show()


    pid_distance_store = np.linalg.norm(pid_res_store[0:2, :] - ref_traj[0:2, :], axis=0)
    plt.figure()
    plt.plot(pid_distance_store)
    plt.title('distance difference')
    plt.legend(['mpc', 'pid'])
    plt.show()

    # plot orientation difference
    orientation_store_pid = np.zeros(mpc.nTraj)
    for i in range(mpc.nTraj):
        X_d = SE2(ref_traj[:, i])
        X_pid = SE2(pid_res_store[:, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        X_d_inv_X_pid = SO2(X_d.angle()).between(SO2(X_pid.angle()))
        orientation_store_pid[i] = scipy.linalg.norm(X_d_inv_X_pid.log().coeffs())

    plt.figure()

    plt.plot(orientation_store_pid[0:])
    plt.title('orientation difference')
    plt.legend(['mpc', 'pid'])

    plt.show()



def test_mpc():
    ref_traj_config = {'start_state': np.array([0, 0, 0]),
                       'linear_vel': 0.5,
                       'angular_vel': 0.5}
    mpc = ErrorDynamicsMPC(ref_traj_config)
    ref_traj, ref_v, dt = mpc.setup_ref_traj(ref_traj_config)
    mpc.setup_solver()
    state = np.array([-0.6, -0.6, 0.1])
    state = SE2Tangent(state).exp().coeffs()
    t = 0
    # contrainer to store state
    state_store = np.zeros((4, mpc.nTraj))
    state_store[:, 0] = state
    pid_res_store = np.zeros((4, mpc.nTraj))
    pid_res_store[:, 0] = state


    # start simulation
    for i in range(mpc.nTraj-1):
        state = state_store[:, i]
        xi = mpc.solve(state, t)
        X = SE2(state)  # SE2 state
        X = X + SE2Tangent(xi * mpc.dt)  # X * SE2Tangent(xi * self.dt).exp()
        state_store[:, i + 1] = X.coeffs()
        t += mpc.dt

    # pid control
    t = 0
    for i in range(mpc.nTraj-1):
        state = pid_res_store[:, i]
        xi = mpc.p_control(state, t)
        X = SE2(state)
        X = X + SE2Tangent(xi * mpc.dt)
        pid_res_store[:, i + 1] = X.coeffs()
        t += mpc.dt


    # plot state
    plt.figure()
    plt.plot(state_store[0, :], state_store[1, :])
    plt.plot(ref_traj[0, :], ref_traj[1, :])
    plt.plot(pid_res_store[0, :], pid_res_store[1, :])
    plt.legend(['state', 'ref_traj', 'pid_res'])
    plt.show()

    # plot distance difference
    distance_store = np.linalg.norm(state_store[0:2, :] - ref_traj[0:2, :], axis=0)
    pid_distance_store = np.linalg.norm(pid_res_store[0:2, :] - ref_traj[0:2, :], axis=0)
    plt.figure()
    plt.plot(distance_store)
    plt.plot(pid_distance_store)
    plt.title('distance difference')
    plt.legend(['mpc', 'pid'])
    plt.show()

    # plot orientation difference
    orientation_store = np.zeros(mpc.nTraj)
    orientation_store_pid = np.zeros(mpc.nTraj)
    for i in range(mpc.nTraj):
        X_d = SE2(ref_traj[:, i])
        X = SE2(state_store[:, i])
        X_pid = SE2(pid_res_store[:, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        X_d_inv_X_pid = SO2(X_d.angle()).between(SO2(X_pid.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())
        orientation_store_pid[i] = scipy.linalg.norm(X_d_inv_X_pid.log().coeffs())

    plt.figure()
    plt.plot(orientation_store[0:])
    plt.plot(orientation_store_pid[0:])
    plt.title('orientation difference')
    plt.legend(['mpc', 'pid'])

    plt.show()




if __name__ == '__main__':
    # test_generate_ref_traj()
   test_pid()
