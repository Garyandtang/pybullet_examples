import time

from controllers.lqr.lqr_utils import *
from envs.turtlebot.turtlebot import Turtlebot
from functools import partial
import numpy as np
import casadi as ca

# class MPC():
#     def __int__(self,env_func, q_lqr: list = None, r_lqr: list = None, **kwargs):
#         self.solver = DirectTransMethod(env_func=env_func, q_lqr=q_lqr, r_lqr=r_lqr)
#
#

# def mpc(env_func, q_lqr: list = None, r_lqr: list = None):
#     solver = DirectTransMethod(env_func=env_func, q_lqr=q_lqr, r_lqr=r_lqr)
#

class DirectTransMethod():
    def __init__(self,
                 env_func,
                 q_lqr: list = None,
                 r_lqr: list = None,
                discrete_dynamics: bool = True,
                **kwargs):
        self.env = env_func()
        self.model = self.get_prior(self.env)
        self.Q = get_cost_weight_matrix(q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(r_lqr, self.model.nu)
        self.discrete_dynamics = discrete_dynamics

        self.curr_state = np.zeros((self.model.nx,1))
        self.goal_state = np.array([1,1,np.pi/2])
        self.T = 15
        self.dt = 0.02

    def set_curr_state(self, curr_state):
        self.curr_state = curr_state

    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    def select_action(self, obs, info=None):
        self.set_curr_state(obs)
        return self.solve()

    def set_mpc_horizon(self):
        dist = np.linalg.norm(self.curr_state[1:2] - self.goal_state[1:2])
        T = int(dist / np.sqrt(2) * self.T)
        return max(min(T, self.T), 3)
    def solve(self):
        nx, nu = self.model.nx, self.model.nu
        T = self.set_mpc_horizon()
        opti = ca.Opti()
        x_var = opti.variable(nx, T+1)
        u_var = opti.variable(nu, T)
        v = u_var[0, :]
        w = u_var[1, :]

        opti.subject_to(x_var[:, 0] == self.curr_state)
        # system model constraints
        for i in range(T):
            # euler method
            x_next = x_var[:, i] + self.dt * self.model.fc_func(x_var[:, i], u_var[:, i])
            opti.subject_to(x_var[:, i+1] == x_next)


        # goal constraint
        opti.subject_to(x_var[:, -1] == self.goal_state)

        # cost
        cost = 0
        cost_func = self.model.cost_func
        for i in range(T):
            # cost += cost_func(np.zeros((nx, 1)),
            #                   np.zeros((nx, 1)),
            #                   u_var[:, i],
            #                   np.zeros((nu, 1)),
            #                   self.Q,
            #                   self.R)
            cost += cost_func(x_var[:, i],
                              self.goal_state,
                              u_var[:, i],
                              np.zeros((nu, 1)),
                              self.Q,
                              self.R)

        # cost += cost_func(x_var[:, -1], self.goal_state, np.zeros((nu, 1)), np.zeros((nu, 1)), 100 * self.Q, self.R)
        opti.minimize(cost)
        opts_setting = {'ipopt.max_iter': 80, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-3,
                        'ipopt.acceptable_obj_change_tol': 1e-3}
        opti.solver('ipopt', opts_setting)

        soln = opti.solve()
        u_sol = soln.value(u_var)
        # x_sol = soln.value(x_var)

        return u_sol[:, 0]






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
    # set env
    env = Turtlebot(gui=True)
    # set solver
    key_word = {'gui': False}
    env_func = partial(Turtlebot, **key_word)
    q_lqr = [10, 10, 10]
    r_lqr = [0.1]
    solver = DirectTransMethod(env_func=env_func, q_lqr=q_lqr, r_lqr=r_lqr)

    while 1:
        curr_state = env.get_state()
        print(curr_state)
        action = solver.select_action(curr_state)
        print(action)
        env.step(action)
        # time.sleep(0.1)