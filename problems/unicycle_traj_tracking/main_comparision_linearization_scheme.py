import time
from environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
import numpy as np
from utils.enum_class import CostType, DynamicsType, TrajType, ControllerType, EnvType, LiniearizationType
from naive_mpc import NaiveMPC
from feedback_linearization import FBLinearizationController
from error_dynamics_mpc import ErrorDynamicsMPC
from ref_traj_generator import TrajGenerator
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from monte_carlo_test_turtlebot import simulation, calulate_trajecotry_error
from matplotlib import pyplot as plt

def main():
    init_state = np.array([-0.2, -0.2, np.pi/3])
    env_type = EnvType.SCOUT_MINI

    traj_config = {'type': TrajType.CIRCLE,
              'param': {'start_state': np.array([0, 0, 0]),
                        'linear_vel': 0.5,
                        'angular_vel': 0.5,
                        'dt': 0.02,
                        'nTraj': 500}}

    controller = ErrorDynamicsMPC(traj_config)
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()

    store_state, store_control, store_solve_time = simulation(init_state, controller, traj_gen, env_type, gui=True)
    controller = ErrorDynamicsMPC(traj_config, linearization_type=LiniearizationType.WEDGE)
    store_state_1, store_control_1, store_solve_time_1 = simulation(init_state, controller, traj_gen, env_type, gui=False)

    position_error, orientation_error = calulate_trajecotry_error(store_state, ref_state)
    position_error_1, orientation_error_1 = calulate_trajecotry_error(store_state_1, ref_state)

    # plot position error
    plt.figure()
    plt.plot(position_error[:], label='position error')
    plt.plot(position_error_1[:], label='position error_1')
    plt.legend()
    plt.show()

    # plot orientation error
    plt.figure()
    plt.plot(orientation_error[:], label='orientation error')
    plt.plot(orientation_error_1[:], label='orientation error_1')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()