import time
from environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
import numpy as np
from utils.enum_class import CostType, DynamicsType, TrajType, ControllerType
from naive_mpc import NaiveMPC
from feedback_linearization import FBLinearizationController
from error_dynamics_mpc import ErrorDynamicsMPC
from ref_traj_generator import TrajGenerator
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent


def main():
    init_x = np.random.uniform(-0.3, 0.3)
    init_y = np.random.uniform(-0.3, 0.3)
    init_theta = np.random.uniform(-np.pi/4, np.pi/4)
    init_state = np.array([0, 0, 0])
    # set env
    env = ScoutMini(gui=True, debug=True, init_state=init_state)
    v_min, v_max, w_min, w_max = env.get_vel_cmd_limit()

    # set solver
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.3,
                             'angular_vel': 0.3,
                             'nTraj': 1000,
                             'dt': 0.02}}


    # figure of eight
    traj_config = {'type': TrajType.EIGHT,
              'param': {'start_state': np.array([0, 0, 0]),
                        'dt': 0.02,
                        'nTraj': 1700}}

    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()

    env.draw_ref_traj(ref_state)
    controller = NaiveMPC(traj_config)

    # controller = FBLinearizationController()
    controller.set_control_bound()
    t = 0
    for i in range(ref_state.shape[1] - 1):
        start = time.time()
        curr_state = env.get_state()
        if controller.controllerType == ControllerType.NMPC:
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.GMPC:
            curr_state = SE2(curr_state[0], curr_state[1], curr_state[2]).coeffs()
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.FEEDBACK_LINEARIZATION:
            curr_ref_state = ref_state[:, i]
            curr_ref_vel_cmd = ref_control[:, i]
            vel_cmd = controller.feedback_control(curr_state, curr_ref_state, curr_ref_vel_cmd)
        print('curr_state: ', curr_state)
        print('xi: ', vel_cmd)
        print('curr_twist:', env.get_twist())

        t += dt
        env.step(env.vel_cmd_to_action(vel_cmd))

if __name__ == '__main__':
    main()