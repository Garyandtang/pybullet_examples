import time
from envs.turtlebot.turtlebot import Turtlebot
from envs.scout.scout_mini import ScoutMini
import numpy as np
from utils.enum_class import CostType, DynamicsType, TrajType, ControllerType, EnvType
from naive_mpc import NaiveMPC
from feedback_linearization import FBLinearizationController
from error_dynamics_mpc import ErrorDynamicsMPC
from ref_traj_generator import TrajGenerator
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from monte_carlo_test_turtlebot import calulate_trajecotry_error, simulation
import os
import matplotlib.pyplot as plt

def butterfly_tracking(env_type, controller_type):
    init_state = np.array([0, 0, 0])
    traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'v_scale': 1,
                             'w_scale': 1,
                             'nTraj': 2500}}

    traj_gen = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = traj_gen.get_traj()
    if controller_type == ControllerType.NMPC:
        controller = NaiveMPC(traj_config)
    elif controller_type == ControllerType.GMPC:
        controller = ErrorDynamicsMPC(traj_config)
    elif controller_type == ControllerType.FEEDBACK_LINEARIZATION:
        controller = FBLinearizationController()

    store_SE2, store_twist = simulation(init_state, controller, traj_gen, env_type, gui=False)

    dir_name = env_type.value + '_' + controller_type.value
    position_error, orientation_error = calulate_trajecotry_error(ref_SE2, store_SE2)
    # mkdir if not exist
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'butterfly_tracking', dir_name)):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'data', 'butterfly_tracking', dir_name))

    data_path = os.path.join(os.path.dirname(__file__), 'data', 'butterfly_tracking', dir_name)
    file_path = os.path.join(data_path, 'position_error.npy')
    np.save(file_path, position_error)
    file_path = os.path.join(data_path, 'orientation_error.npy')
    np.save(file_path, orientation_error)
    file_path = os.path.join(data_path, 'ref_SE2.npy')
    np.save(file_path, ref_SE2)
    file_path = os.path.join(data_path, 'store_SE2.npy')
    np.save(file_path, store_SE2)
    file_path = os.path.join(data_path, 'store_twist.npy')
    np.save(file_path, store_twist)
    file_path = os.path.join(data_path, 'ref_twist.npy')
    np.save(file_path, ref_twist)


def main():
    for env_type in EnvType:
        for controller_type in ControllerType:
            print('env_type: {}, controller_type: {}'.format(env_type.value, controller_type.value))
            butterfly_tracking(env_type, controller_type)


if __name__ == '__main__':
    main()
