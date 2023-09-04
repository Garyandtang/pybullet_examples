import time

from controllers.lqr.lqr_utils import *
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from environments.wheeled_mobile_robot.turtlebot.turtlebot_model import TurtlebotModel
from functools import partial
from controllers.dtm.direct_trans_method import DirectTransMethod
import numpy as np
import casadi as ca
from utils.enum_class import CostType, DynamicsType
class turtle_move_to_problem():
    def __init__(self):
        config = {'cost_type': CostType.POSITION_QUATERNION,
                  'dynamics_type': DynamicsType.EULER_FIRST_ORDER}
        key_word = {'gui': False, 'config': config}
        # dynamics
        self.model = TurtlebotModel(config=config).symbolic
        self.nx = self.model.nx
        self.nu = self.model.nu

        # cost function
        q_lqr = [10, 10, 2]
        r_lqr = [0.1]
        self.set_cost_weights(q_lqr, r_lqr)

        # constraints
        # boundary constraints
        self.x_start = np.zeros((self.nx, 1))
        self.x_goal = np.zeros((self.nx, 1))

        self.u_upper = np.ones((self.nu, 1)) * 20
        self.u_lower = np.ones((self.nu, 1)) * -20

        # horizon
        self.T = 0.1
        self.dt = self.model.dt  # 0.02s
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