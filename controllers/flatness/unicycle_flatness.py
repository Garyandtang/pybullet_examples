import casadi as ca
from matplotlib import pyplot as plt
import numpy as np
from envs.turtlebot.turtlebot import Turtlebot
from envs.turtlebot.turtlebot_model import TurtlebotModel
from functools import partial
from controllers.dtm.direct_trans_method import DirectTransMethod
import time
from utils.enum_class import CostType, DynamicsType

"""
This file contains the implementation of the flatness controller for the unicycle model.
unicycle model:
state:
    x: x position
    y: y position
    theta: orientation
control:
    v: linear velocity
    w: angular velocity
dynamics:
    x_dot = v * cos(theta)
    y_dot = v * sin(theta)
    theta_dot = w
flatness output:
    x = z1
    y = z2
    theta = = arctan(z1_dot/z2_dot)
    
    v = sqrt(z1_dot^2 + z2_dot^2)
    w = (z1_ddot * z2_dot - z1_dot * z2_ddot) / (z1_dot^2 + z2_dot^2)
    
implementation
    z1 = a0 * t^3 + a1 * t^2 + a2 * t + a3
    z2 = b0 * t^3 + b1 * t^2 + b2 * t + b3
"""

class UnicycleFlatnessController:
    def __init__(self):
        self.name = 'Unicycle Flatness Controller'
        self.nFlat = 2
        self.order = 4  # order of the flatness output


    def setup_coefficient_solver(self):
        t0 = ca.MX.sym('t0')
        tf = ca.MX.sym('tf')
        t = ca.vertcat(t0, tf)
        A = ca.vertcat(ca.horzcat(t0 ** 3, t0 ** 2, t0, 1),     # init pos
                       ca.horzcat(tf ** 3, tf ** 2, tf, 1),     # final pos
                       ca.horzcat(3 * t0 ** 2, 2 * t0, 1, 0),   # init vel
                       ca.horzcat(3 * tf ** 2, 2 * tf, 1, 0))   # final vel
        z0 = ca.MX.sym('z0')
        z0_dot = ca.MX.sym('z0_dot')
        zf = ca.MX.sym('zf')
        zf_dot = ca.MX.sym('zf_dot')
        b = ca.vertcat(z0, zf, z0_dot, zf_dot)

        # x = ca.MX.sym('x', 4)


        a = ca.solve(A, b)
        self.coef_func = ca.Function('a_func', [t, b], [a])

    def compute_flatness_coeff(self, x0, xf, tf):
        x0 = [0, 1, 0, 0]
        xf = [0, 1, 0, 0]
        t = [0, 1]
        a0 = self.coef_func(t, x0)
        a1 = self.coef_func(t, xf)
        return a0, a1

    def setup_flatness_output(self):
        t = ca.MX.sym('t')
        a = ca.MX.sym('a', 4)
        b = ca.MX.sym('b', 4)
        z1 = a[0] * t ** 3 + a[1] * t ** 2 + a[2] * t + a[3]
        z2 = b[0] * t ** 3 + b[1] * t ** 2 + b[2] * t + b[3]
        z1_dot = ca.jacobian(z1, t)
        z2_dot = ca.jacobian(z2, t)
        z1_ddot = ca.jacobian(z1_dot, t)
        z2_ddot = ca.jacobian(z2_dot, t)
        z = ca.vertcat(z1, z2)
        z_dot = ca.vertcat(z1_dot, z2_dot)
        z_ddot = ca.vertcat(z1_ddot, z2_ddot)
        flatness_output = {'z': z, 'z_dot': z_dot, 'z_ddot': z_ddot, 'var': {'a': a, 'b': b, 't': t}}
        return flatness_output



def main():
    waypoints = np.array([[0, 0, 0], [1, 1, np.pi / 2], [2, -1, -np.pi / 2], [3, 1, np.pi / 2]])

    waypoints = np.array([[0, 0, 0], [1, 1, np.pi / 2], [0, 2, np.pi], [-1, 1, -np.pi / 2]])

    waypoints = np.array([[1, 1, np.pi]])
    # set env
    env = Turtlebot(gui=True)
    # set solver
    key_word = {'gui': False}
    env_func = partial(Turtlebot, **key_word)
    problem = turtle_move_to_problem()
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


if __name__ == '__main__':
    main()