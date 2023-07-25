import time

from controllers.lqr.lqr_utils import *
from envs.turtlebot.turtlebot import Turtlebot
from envs.turtlebot.turtlebot_model import TurtlebotModel
from functools import partial
import numpy as np
import casadi as ca
from utils.enum_class import CostType, DynamicsType
class DirectTransMethod():
    def __init__(self,
                 problem,
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


