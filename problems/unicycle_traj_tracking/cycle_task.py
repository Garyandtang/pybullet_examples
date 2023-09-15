import time
from environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
import numpy as np
from utils.enum_class import CostType, DynamicsType, TrajType, ControllerType, EnvType
from naive_mpc import NaiveMPC
from feedback_linearization import FBLinearizationController
from error_dynamics_mpc import ErrorDynamicsMPC
from ref_traj_generator import TrajGenerator
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from monte_carlo_test_turtlebot import simulation, calulate_trajecotry_error
from matplotlib import pyplot as plt

def main():
    init_state = np.array([0, 0, 0])
    controller_type = ControllerType.NMPC
    env_type = EnvType.TURTLEBOT
    # set solver
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.3,
                             'angular_vel': 0.3,
                             'nTraj': 200,
                             'dt': 0.02}}

    # # figure of eight
    # traj_config = {'type': TrajType.EIGHT,
    #           'param': {'start_state': np.array([0, 0, 0]),
    #                     'dt': 0.02,
    #                     'nTraj': 1700}}

    if controller_type == ControllerType.NMPC:
        controller = NaiveMPC(traj_config)
    elif controller_type == ControllerType.GMPC:
        controller = ErrorDynamicsMPC(traj_config)
    elif controller_type == ControllerType.FEEDBACK_LINEARIZATION:
        controller = FBLinearizationController()

    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()

    store_state, store_control, store_solve_time = simulation(init_state, controller, traj_gen, env_type, gui=False)

    # plot linear velocity
    plt.figure()
    plt.plot(store_control[0, :], label='linear velocity')
    plt.plot(store_control[1, :], label='angular velocity')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()