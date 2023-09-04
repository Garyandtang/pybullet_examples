import time

from controllers.lqr.lqr_utils import *
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from environments.wheeled_mobile_robot.turtlebot.turtlebot_model import TurtlebotModel
from functools import partial
from controllers.dtm.direct_trans_method import DirectTransMethod
import numpy as np
import casadi as ca
from utils.enum_class import CostType, DynamicsType
def main():

    # set env
    env = Turtlebot(gui=True)
    # set solver


    while 1:
        start = time.time()
        curr_state = env.get_state()
        print('curr_state: ', curr_state)
        vel_cmd = np.array([0.5, 0.3])
        action = env.vel_cmd_to_action(vel_cmd)
        print('action: ', action)
        end = time.time()
        print('time: ', end - start)
        env.step(action)

        # time.sleep(0.1)


if __name__ == '__main__':
    main()