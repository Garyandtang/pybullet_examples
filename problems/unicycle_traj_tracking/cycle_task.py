import time

from controllers.lqr.lqr_utils import *
from envs.turtlebot.turtlebot import Turtlebot
from envs.turtlebot.turtlebot_model import TurtlebotModel
from functools import partial
from controllers.dtm.direct_trans_method import DirectTransMethod
import numpy as np
import casadi as ca
from utils.enum_class import CostType, DynamicsType, TrajType
from error_dynamics_mpc import ErrorDynamicsMPC
from ref_traj_generator import TrajGenerator
from manifpy import SE2, SE2Tangent
def main():

    # set env
    env = Turtlebot(gui=True)

    # set solver
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 670,
                             'dt': 0.02}}
    mpc = ErrorDynamicsMPC(traj_config)
    dt = mpc.dt
    t = 0
    while 1:
        start = time.time()
        curr_state = env.get_state()
        # convert state to SE2
        curr_state = SE2(curr_state[0], curr_state[1], curr_state[2]).coeffs()
        print('curr_state: ', curr_state)
        xi = mpc.solve(curr_state, t)
        print('xi: ', xi)
        t += dt
        env.step(env.vel_cmd_to_action(xi[0:2]))

if __name__ == '__main__':
    main()