import os
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from controller.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType, ControllerType, LiniearizationType
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from environments.numerical_simulator.WMR_simulator import WMRSimulator
from data_driven_FBC import LTI, calculate_r_l
import casadi as ca
class time_varying_LQR:
    def __init__(self, traj_config, lti):
        self.system_init(traj_config, lti)
        self.controller_init()
        pass

    def system_init(self, traj_config, lti):
        self.Q = lti.Q
        self.R = lti.R

        tran_gen = TrajGenerator(traj_config)
        ref_state, ref_control, dt = tran_gen.get_traj()
        self.ref_state = ref_state
        self.ref_control = ref_control # [v, w]
        self.nTraj = ref_state.shape[1]

        B_list = [lti.B for i in range(self.nTraj)]
        self.B_list = np.array(B_list)
        A_list = []
        for i in range(self.nTraj):
            vel_cmd = ref_control[:, i]
            twist = np.array([vel_cmd[0], 0, vel_cmd[1]])
            adj = -SE2Tangent(twist).smallAdj()
            A = np.eye(3) + dt * adj
            A_list.append(A)
        self.A_list = np.array(A_list)
        self.Q_list = [self.Q for i in range(self.nTraj)]
        self.R_list = [self.R for i in range(self.nTraj)]
        self.dt = dt


    def controller_init(self):
        self.K_list = []
        self.k_list = []
        # solve with backward
        P = self.Q
        for i in range(self.nTraj - 1, -1, -1):
            A = self.A_list[i]
            B = self.B_list[i]
            Q = self.Q_list[i]
            R = self.R_list[i]
            vel_cmd = self.ref_control[:, i]
            twist = np.array([vel_cmd[0], 0, vel_cmd[1]]) * self.dt

            K = -ct.dlqr(A, B, Q, R)[0]
            self.K_list.append(K)
            k = np.linalg.pinv(B) @ twist
            self.k_list.append(k)
            P = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

    def step(self, curr_state, i):
        ref_state = self.ref_state[:, i]
        x = SE2(ref_state[0], ref_state[1], ref_state[2]).between(SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
        u = self.K_list[i] @ x + self.k_list[i]
        return u

    def mpc_step(self, curr_state, k):
        ref_state = self.ref_state[:, k]
        x_init = SE2(ref_state[0], ref_state[1], ref_state[2]).between(SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
        Q = self.Q
        R = self.R
        N = 10
        dt = self.dt
        # setup casadi solver
        opti = ca.Opti('conic')
        # opti = ca.Opti()
        x_var = opti.variable(3, N + 1)
        u_var = opti.variable(2, N)

        # setup initial condition
        opti.subject_to(x_var[:, 0] == x_init)

        # setup dynamics constraints
        # x_next = A * x + B * u + h
        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            ref_v = self.ref_control[:, index]  # desir
            ref_twist = np.array([ref_v[0], 0, ref_v[1]])
            u_d = np.linalg.pinv(self.B_list[index]) @ ref_twist * dt
            A = self.A_list[index]
            B = self.B_list[index]
            x_next = A @ x_var[:, i] + B @ (u_var[:, i] - u_d)
            opti.subject_to(x_var[:, i + 1] == x_next)


        # cost function
        cost = 0
        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            ref_v = self.ref_control[:, index]  # desir
            ref_twist = np.array([ref_v[0], 0, ref_v[1]])
            u_d = np.linalg.pinv(self.B_list[index]) @ ref_twist * dt
            cost += ca.mtimes([x_var[:, i].T, Q, x_var[:, i]]) + ca.mtimes(
                [(u_var[:, i] - u_d).T, R, (u_var[:, i] - u_d)])

        cost += ca.mtimes([x_var[:, N].T, 100 * Q, x_var[:, N]])
        opts_setting = {'printLevel': 'none'}
        opti.minimize(cost)
        opti.solver('qpoases', opts_setting)
        sol = opti.solve()
        psi_sol = sol.value(x_var)
        u_sol = sol.value(u_var)
        return u_sol[:, 0]

if __name__ == '__main__':
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.zeros((3,)),
                             'linear_vel': 0.02,
                             'angular_vel': 0.2,
                             'nTraj': 17,
                             'dt': 0.02}}
    traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'scale': 2,
                             'nTraj': 2500}}

    lti = LTI()
    lqr = time_varying_LQR(traj_config, lti)
    state_container = np.zeros((3, lqr.nTraj-1))
    x_container = np.zeros((3, lqr.nTraj-1))
    ref_state_container = lqr.ref_state
    robot = WMRSimulator()
    for i in range(lqr.nTraj-1):
        curr_state = robot.get_state()
        state_container[:, i] = curr_state
        u = lqr.mpc_step(curr_state, i)
        next_state, _, _, _, _ = robot.step(u)
        x = SE2(ref_state_container[0, i], ref_state_container[1, i], ref_state_container[2, i]).between(
            SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
        x_container[:, i] = x

    # plot trajectory
    font_size = 12
    line_width = 2
    plt.figure()
    plt.grid(True)
    plt.plot(ref_state_container[0, :lqr.nTraj - 1], ref_state_container[1, :lqr.nTraj - 1], 'b')
    plt.plot(state_container[0, :], state_container[1, :], 'r')
    plt.xlabel("$x~(m)$", fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)
    name = "trained_trajectory.jpg"
    plt.show()

    # plot x
    plt.figure()
    plt.grid(True)
    plt.plot(x_container.T)
    plt.ylim([-0.15, 0.15])
    plt.xlabel("k", fontsize=font_size)
    plt.ylabel("$x$", fontsize=font_size)
    name = "trained_x.jpg"
    plt.show()

    # # plot init x and y in the same figure
    # plt.figure()
    # plt.grid(True)
    #
    # font_size = 12
    # line_width = 2
    # plt.xticks(fontsize=font_size - 2)
    # plt.yticks(fontsize=font_size - 2)
    # plt.plot(lqr.ref_state[0, :], lqr.ref_state[1, :], linewidth=line_width)
    # # plt.tight_layout()
    # plt.xlabel("$x~(m)$", fontsize=font_size)
    # plt.ylabel("$y~(m)$", fontsize=font_size)
    # name = "init_trajectory.jpg"
    #
    # plt.show()
    # pass