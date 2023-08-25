import time
from envs.turtlebot.turtlebot import Turtlebot
from envs.scout.scout_mini import ScoutMini
import numpy as np
from utils.enum_class import CostType, DynamicsType, TrajType, ControllerType
from naive_mpc import NaiveMPC
from feedback_linearization import FBLinearizationController
from error_dynamics_mpc import ErrorDynamicsMPC
from ref_traj_generator import TrajGenerator
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
def main():

    # set env
    env = ScoutMini(gui=True)
    v_min, v_max, w_min, w_max = env.get_vel_cmd_limit()

    # set solver
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([1, 1, 0]),
                             'linear_vel': 0.1,
                             'angular_vel': 0.1,
                             'nTraj': 6700,
                             'dt': 0.02}}

    # figure of eight
    traj_config = {'type': TrajType.EIGHT,
              'param': {'start_state': np.array([0, 0, 0]),
                        'dt': 0.02,
                        'v_scale': 0.5,
                        'w_scale': 1,
                        'nTraj': 2500}}

    traj_gen = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = traj_gen.get_traj()
    v_min_, v_max_, w_min_, w_max_ = traj_gen.get_vel_bound()
    print("v_min: ", v_min_, "v_max: ", v_max_, "w_min: ", w_min_, "w_max: ", w_max_)
    env.draw_ref_traj(ref_SE2)
    controller = ErrorDynamicsMPC(traj_config)
    controller.set_control_bound(v_min, v_max, w_min, w_max)
    # controller = FBLinearizationController()
    t = 0
    for i in range(ref_SE2.shape[1] - 1):
        start = time.time()
        curr_state = env.get_state()
        if controller.controllerType == ControllerType.NMPC:
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.GMPC:
            curr_state = SE2(curr_state[0], curr_state[1], curr_state[2]).coeffs()
            vel_cmd = controller.solve(curr_state, t)
        elif controller.controllerType == ControllerType.FEEDBACK_LINEARIZATION:
            curr_ref_SE2 = ref_SE2[:, i]
            curr_ref_X = SE2(curr_ref_SE2)
            curr_ref_state = np.array([curr_ref_X.x(), curr_ref_X.y(), curr_ref_X.angle()])
            curr_ref_twist = ref_twist[:, i]
            curr_ref_vel_cmd = np.array([curr_ref_twist[0], curr_ref_twist[2]])
            vel_cmd = controller.feedback_control(curr_state, curr_ref_state, curr_ref_vel_cmd)
        print('curr_state: ', curr_state)
        print('xi: ', vel_cmd)
        print('curr_twist:', env.get_twist())

        t += dt
        env.step(env.vel_cmd_to_action(vel_cmd))

if __name__ == '__main__':
    main()