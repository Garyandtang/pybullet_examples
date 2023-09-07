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
    # set init state (real experiment should get from motion capture system)
    init_state = np.array([0, 0, 0])
    # set ref trajectory
    traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'v_scale': 1.8,
                             'w_scale': 1, # don't change this
                             'nTraj': 2500}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    # set environment
    env = ScoutMini(gui=True, debug=True, init_state=init_state)
    env.draw_ref_traj(ref_state)

    # set controller
    ctrl_type = ControllerType.NMPC
    if ctrl_type == ControllerType.NMPC:
        controller = NaiveMPC(traj_config)
    elif ctrl_type == ControllerType.GMPC:
        controller = ErrorDynamicsMPC(traj_config)
    controller.set_control_bound()
    t = 0
    store_solve_time = np.zeros(ref_state.shape[1]-1)
    for i in range(ref_state.shape[1] - 1):
        curr_state = env.get_state()
        if controller.controllerType == ControllerType.NMPC:
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.GMPC:
            curr_state = SE2(curr_state[0], curr_state[1], curr_state[2]).coeffs()
            vel_cmd = controller.solve(curr_state, t)
        store_solve_time[i] = controller.get_solve_time()
        print('curr_state: ', curr_state)
        print('xi: ', vel_cmd)
        print('curr_twist:', env.get_twist())

        t += dt
        env.step(env.vel_cmd_to_action(vel_cmd))

    np.save(ctrl_type.value + '_' + 'store_solve_time.npy', store_solve_time)

if __name__ == '__main__':
    main()