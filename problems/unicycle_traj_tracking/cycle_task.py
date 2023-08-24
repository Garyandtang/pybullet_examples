import time
from envs.turtlebot.turtlebot import Turtlebot
from envs.scout.scout_mini import ScoutMini
import numpy as np
from utils.enum_class import CostType, DynamicsType, TrajType
from naive_mpc import NaiveMPC
from error_dynamics_mpc import ErrorDynamicsMPC
from ref_traj_generator import TrajGenerator
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
def main():

    # set env
    env = Turtlebot(gui=True)

    # set solver
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.1,
                             'nTraj': 670,
                             'dt': 0.02}}

    # figure of eight
    traj_config = {'type': TrajType.EIGHT,
              'param': {'start_state': np.array([0, 0, 0]),
                        'dt': 0.02,
                        'scale': 1.5,
                        'nTraj': 3000}}

    traj_gen = TrajGenerator(traj_config)
    ref_traj, ref_v, dt = traj_gen.get_traj()
    env.draw_ref_traj(ref_traj)
    mpc = NaiveMPC(traj_config)
    dt = mpc.dt
    t = 0
    while 1:
        start = time.time()
        curr_state = env.get_state()
        # convert state to SE2
        # if use errorDynamicsMPC, then uncommented the following line
        # curr_state = SE2(curr_state[0], curr_state[1], curr_state[2]).coeffs()
        print('curr_state: ', curr_state)
        print('ref_state: ', mpc.get_curr_ref(t)[0])
        xi = mpc.solve(curr_state, t)
        print('xi: ', xi)
        print('curr_twist:', env.get_twist())

        t += dt
        env.step(env.vel_cmd_to_action(xi[0:2]))

if __name__ == '__main__':
    main()