import time

from controllers.lqr.lqr_utils import *
from envs.turtlebot.turtlebot import Turtlebot
from functools import partial
import numpy as np
import casadi as ca
from utils.enum_class import CostType, DynamicsType


class problem():
    def __init__(self, env_func):
        config = {'cost_type': CostType.POSITION_EULER, 'dynamics_type': DynamicsType.NORMAL_FIRST_ORDER}
        key_word = {'gui': False, 'config': config}
        env_func = partial(Turtlebot, **key_word)
        # dynamics
        self.env = env_func()
        self.model = self.env.symbolic
        self.nx = self.model.nx
        self.nu = self.model.nu

        # cost function
        # default cost weights
        self.q_lqr = np.ones((self.nx, 1))
        self.r_lqr = np.ones((self.nu, 1))
        self.cost_func = self.model.cost_func

        # constraints
        # boundary constraints
        self.x_start = np.zeros((self.nx, 1))
        self.x_goal = np.zeros((self.nx, 1))

        self.u_upper = np.ones((self.nu, 1)) * 20
        self.u_lower = np.ones((self.nu, 1)) * -20

        # horizon
        self.T = 0.1
        self.dt = self.env.CTRL_TIMESTEP  # 50Hz
        self.N = int(self.T / self.dt)

        self.set_discrete_dynamics()

    def set_predict_horizon(self, T):
        self.T = T
        self.N = int(self.T / self.dt)

    def set_cost_weights(self, q_lqr, r_lqr):
        self.q_lqr = q_lqr
        self.r_lqr = r_lqr
        self.cost_func = self.model.cost_func

    def set_start_state(self, x_start):
        self.x_start = x_start

    def set_goal_state(self, x_goal):
        self.x_goal = x_goal

    def set_boundary(self, x_start, x_goal):
        self.x_start = x_start
        self.x_goal = x_goal

    def set_discrete_dynamics(self):
        l = 0.025
        x, y, theta = ca.MX.sym('x'), ca.MX.sym('y'), ca.MX.sym('theta')
        v_l, v_r = ca.MX.sym('v_l'), ca.MX.sym('v_r')
        v = (v_l + v_r) / 2
        w = (v_r - v_l) / l
        x_next = x + v * ca.cos(theta) * self.dt
        y_next = y + v * ca.sin(theta) * self.dt
        theta_next = theta + w * self.dt
        X_next_1 = ca.vertcat(x_next, y_next, theta_next)
        X = ca.vertcat(x, y, theta)
        x_next = x + v*(ca.sin(theta + w * self.dt) - ca.sin(theta))/w
        y_next = y - v*(ca.cos(theta + w * self.dt) - ca.cos(theta))/w
        theta_next = theta + w * self.dt
        X_next_2 = ca.vertcat(x_next, y_next, theta_next)
        X_next = ca.if_else(ca.fabs(w) <= 1e-3, X_next_1, X_next_2)
        self.fd_func = ca.Function('fd_func', [X, ca.vertcat(v_l, v_r)], [X_next])


class DirectTransMethod():
    def __init__(self,
                 problem: problem,
                 **kwargs):
        # parse dynamics
        self.model = problem.model
        self.nx = self.model.nx
        self.nu = self.model.nu
        # parse cost function
        self.Q = get_cost_weight_matrix(problem.q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(problem.r_lqr, self.model.nu)
        self.cost_func = problem.cost_func
        # parse constraints
        self.x_start = problem.x_start
        self.x_goal = problem.x_goal
        self.u_upper = problem.u_upper
        self.u_lower = problem.u_lower
        # parse horizon
        self.N = problem.N
        self.dt = problem.dt
        self.fd_func = problem.fd_func

    def solve(self):
        nx, nu = self.model.nx, self.model.nu
        T = self.set_mpc_horizon()
        opti = ca.Opti()
        x_var = opti.variable(nx, T + 1)
        u_var = opti.variable(nu, T)
        v = u_var[0, :]
        w = u_var[1, :]

        opti.subject_to(x_var[:, 0] == self.x_start)

        # control bound constraints
        for i in range(T):
            opti.subject_to(self.u_lower <= u_var[:, i])
            opti.subject_to(u_var[:, i] <= self.u_upper)


        # system model constraints
        for i in range(T):
            # euler method
            x_next = x_var[:, i] + self.dt * self.model.fc_func(x_var[:, i], u_var[:, i])
            # x_next = self.fd_func(x_var[:, i], u_var[:, i])
            opti.subject_to(x_var[:, i + 1] == x_next)

        # goal constraint
        # opti.subject_to(x_var[:, -1] == self.x_goal)

        # cost
        cost = 0
        cost_func = self.cost_func
        for i in range(T):
            cost += cost_func(x_var[:, i], self.x_goal, u_var[:, i], np.zeros((nu, 1)), self.Q, self.R)

        cost += cost_func(x_var[:, -1], self.x_goal, np.zeros((nu, 1)), np.zeros((nu, 1)), 100 * self.Q, self.R)
        opti.minimize(cost)
        opts_setting = {'ipopt.max_iter': 80, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-3,
                        'ipopt.acceptable_obj_change_tol': 1e-3}
        opti.solver('ipopt', opts_setting)

        soln = opti.solve()
        u_sol = soln.value(u_var)
        # x_sol = soln.value(x_var)

        return u_sol[:, 0]

    def set_mpc_horizon(self):
        dist = np.linalg.norm(self.x_start[0:1] - self.x_goal[0:1])
        N = int(dist / np.sqrt(2) * self.N)
        return max(min(N, self.N), 2)





    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def get_prior(self, env):
        return env.symbolic

    def setup_optimizer(self):
        raise NotImplementedError

    def compute_init_guess(self):
        raise NotImplementedError


if __name__ == '__main__':
    # trajectory
    waypoints = np.array([[0, 0, 0], [1, 1, np.pi/2], [2, -1, -np.pi/2], [3, 1, np.pi/2]])

    waypoints = np.array([[0, 0, 0], [1, 1, np.pi / 2], [0, 2, np.pi], [-1, 1, -np.pi / 2]])

    waypoints = np.array([[1, 1, np.pi]])
    # set env
    env = Turtlebot(gui=True)
    # set solver
    key_word = {'gui': False}
    env_func = partial(Turtlebot, **key_word)
    problem = problem(env_func)
    q_lqr = [10, 10, 2]
    r_lqr = [0.1]
    problem.set_cost_weights(q_lqr, r_lqr)
    index = 0
    goal_state = waypoints[index, :]
    problem.set_goal_state(goal_state)
    solver = DirectTransMethod(problem)

    while 1:
        start = time.time()
        curr_state = env.get_state()
        print('curr_state: ', curr_state)
        if np.linalg.norm(curr_state[0:2] - goal_state[0:2]) < 0.01:
            index += 0
            goal_state = waypoints[index % 4, :]
            problem.set_goal_state(goal_state)

        problem.set_start_state(curr_state)
        solver = DirectTransMethod(problem)
        action = solver.solve()
        print('action: ', action)
        end = time.time()
        print('time: ', end - start)
        env.step(action)

        # time.sleep(0.1)
