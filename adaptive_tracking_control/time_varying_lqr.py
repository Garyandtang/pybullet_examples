import os
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from planner.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType, ControllerType, LiniearizationType
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from environments.numerical_simulator.WMR_simulator import WMRSimulator
from data_driven_FBC import LTI, calculate_r_l
from controller.feedback_linearization import FBLinearizationController
import casadi as ca
class time_varying_LQR:
    def __init__(self, traj_config, lti, r, l):
        self.system_init(traj_config, lti, r, l)

    def system_init(self, traj_config, lti, r, l):
        self.Q = lti.Q
        self.R = lti.R

        tran_gen = TrajGenerator(traj_config)
        ref_state, ref_control, dt = tran_gen.get_traj()
        self.ref_state = ref_state
        self.ref_control = ref_control # [v, w]
        self.nTraj = ref_state.shape[1]
        B = dt * np.array([[r / 2, r / 2],
                            [0, 0],
                            [-r / l, r / l]])
        B_list = [B for i in range(self.nTraj)]
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

def time_varying_simulation(r, l):
    traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'scale': 2,
                             'nTraj': 2500}}

    lti = LTI()
    lqr = time_varying_LQR(traj_config, lti, r, l)
    state_container = np.zeros((3, lqr.nTraj - 1))
    x_container = np.zeros((3, lqr.nTraj - 1))
    ref_state_container = lqr.ref_state
    robot = WMRSimulator()
    robot.set_init_state(np.array([0.1, -0.1, np.pi / 6]))
    vel_container = np.zeros((2, lqr.nTraj - 1))
    for i in range(lqr.nTraj - 1):
        curr_state = robot.get_state()
        state_container[:, i] = curr_state
        u = lqr.mpc_step(curr_state, i)
        vel = robot.control_to_twist(u)
        vel = np.array([vel[0], vel[2]])
        vel_container[:, i] = vel
        next_state, _, _, _, _ = robot.step(u)
        x = SE2(ref_state_container[0, i], ref_state_container[1, i], ref_state_container[2, i]).between(
            SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
        x_container[:, i] = x
    return state_container, vel_container, lqr.ref_state, x_container, lqr.ref_control
def time_varying_simulation_FBC(r, l):
    traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'scale': 2,
                             'nTraj': 2500}}

    lti = LTI()
    lqr = time_varying_LQR(traj_config, lti, r, l)
    fbc = FBLinearizationController()

    state_container = np.zeros((3, lqr.nTraj - 1))
    x_container = np.zeros((3, lqr.nTraj - 1))
    ref_state_container = lqr.ref_state
    robot = WMRSimulator()
    robot.set_init_state(np.array([0.1, -0.1, np.pi / 6]))
    vel_container = np.zeros((2, lqr.nTraj - 1))
    for i in range(lqr.nTraj - 1):
        curr_state = robot.get_state()
        state_container[:, i] = curr_state
        # u = lqr.mpc_step(curr_state, i)
        u = fbc.feedback_control(curr_state, lqr.ref_state[:, i], lqr.ref_control[:, i])
        u = fbc.vel_cmd_to_wheel_vel(u, r, l)
        vel = robot.control_to_twist(u)
        vel = np.array([vel[0], vel[2]])
        vel_container[:, i] = vel
        next_state, _, _, _, _ = robot.step(u)
        x = SE2(ref_state_container[0, i], ref_state_container[1, i], ref_state_container[2, i]).between(
            SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
        x_container[:, i] = x
    return state_container, vel_container, lqr.ref_state, x_container, lqr.ref_control

if __name__ == '__main__':

    init_state_container, init_vel_container, ref_state_container, init_x_container, ref_vel = time_varying_simulation(0.03, 0.3)
    trained_state_container, trained_vel_container, trained_ref_state_container, x_container, ref_vel = time_varying_simulation(0.036, 0.23)
    fbc_state_container, fbc_vel_container, _, _, _ = time_varying_simulation_FBC(0.03, 0.3)
    # get curr dir
    curr_dir = os.getcwd()
    save_dir = os.path.join(curr_dir, 'data', 'time_varying_tracking')
    font_size = 15
    line_width = 2

    # plot trajectory
    # initial model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(init_state_container[0, :], init_state_container[1, :])
    # plt.plot(trained_state_container[0, :], trained_state_container[1, :])
    plt.plot(ref_state_container[0, :], ref_state_container[1, :])
    plt.legend(['trajectory with initial MPC', 'reference trajectory'], fontsize=font_size - 2)
    plt.xlabel("$x~(m)$", fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)
    name = "init_time_vary_trajectory.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()

    # trained model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(trained_state_container[0, :], trained_state_container[1, :])
    plt.plot(ref_state_container[0, :], ref_state_container[1, :])
    plt.legend(['trajectory with learned MPC', 'reference trajectory'], fontsize=font_size - 2)
    plt.xlabel("$x~(m)$", fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)
    name = "learned_time_vary_trajectory.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()

    # fbc model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(fbc_state_container[0, :], fbc_state_container[1, :])
    plt.plot(ref_state_container[0, :], ref_state_container[1, :])
    plt.legend(['trajectory with FBC', 'reference trajectory'], fontsize=font_size - 2)
    plt.xlabel("$x~(m)$", fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)
    name = "fbc_time_vary_trajectory.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()


    # plot vel[0]
    # initial model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(init_vel_container[0, :])
    # plt.plot(trained_vel_container[0, :])
    plt.plot(ref_vel[0, :])
    plt.legend(['$v$ of initial MPC', 'reference $v$'], fontsize=font_size - 2)
    plt.xlabel("k", fontsize=font_size)
    plt.ylabel("$v~(m/s)$", fontsize=font_size)
    name = "init_time_vary_v.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()

    # learned model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    # plt.plot(init_vel_container[0, :])
    plt.plot(trained_vel_container[0, :])
    plt.plot(ref_vel[0, :])
    plt.legend(['$v$ of learned MPC', 'reference $v$'], fontsize=font_size - 2)
    plt.xlabel("k", fontsize=font_size)
    plt.ylabel("$v~(m/s)$", fontsize=font_size)
    name = "learned_time_vary_v.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()

    # fbc model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    # plt.plot(init_vel_container[0, :])
    plt.plot(fbc_vel_container[0, :])
    plt.plot(ref_vel[0, :])
    plt.legend(['$v$ of FBC', 'reference $v$'], fontsize=font_size - 2)
    plt.xlabel("k", fontsize=font_size)
    plt.ylabel("$v~(m/s)$", fontsize=font_size)
    name = "fbc_time_vary_v.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()



    # plot vel[1]
    # initial model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(init_vel_container[1, :])
    # plt.plot(trained_vel_container[1, :])
    plt.plot(ref_vel[1, :])
    plt.legend(['$w$ of initial MPC', 'reference $w$'], fontsize=font_size - 2)
    plt.xlabel("k", fontsize=font_size)
    plt.ylabel("$w~(rad/s)$", fontsize=font_size)
    name = "init_time_vary_w.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()

    # learned model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    # plt.plot(init_vel_container[1, :])
    plt.plot(trained_vel_container[1, :])
    plt.plot(ref_vel[1, :])
    plt.legend(['$w$ of learned MPC', 'reference $w$'], fontsize=font_size - 2)
    plt.xlabel("k", fontsize=font_size)
    plt.ylabel("$w~(rad/s)$", fontsize=font_size)
    name = "learned_time_vary_w.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()

    # fbc model
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    # plt.plot(init_vel_container[1, :])
    plt.plot(fbc_vel_container[1, :])
    plt.plot(ref_vel[1, :])
    plt.legend(['$w$ of FBC', 'reference $w$'], fontsize=font_size - 2)
    plt.xlabel("k", fontsize=font_size)
    plt.ylabel("$w~(rad/s)$", fontsize=font_size)
    name = "fbc_time_vary_w.jpg"
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path)
    plt.show()


