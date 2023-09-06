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

def single_tracking_task():
    env_type = EnvType.TURTLEBOT
    controller_type = ControllerType.GMPC
    init_state = np.array([0, 0, 0])
    traj_config = {'type': TrajType.EIGHT,
              'param': {'start_state': np.array([1, 1, 0]),
                        'dt': 0.02,
                        'scale': 1,
                        'nTraj': 300}}
    traj_gen = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = traj_gen.get_traj()
    if controller_type == ControllerType.NMPC:
        controller = NaiveMPC(traj_config)
    elif controller_type == ControllerType.GMPC:
        controller = ErrorDynamicsMPC(traj_config)
    elif controller_type == ControllerType.FEEDBACK_LINEARIZATION:
        controller = FBLinearizationController()

    store_SE2, store_twist, store_solve_time = simulation(init_state, controller, traj_gen, env_type, gui=False)

    # quartile chart
    plt.figure()
    plt.boxplot(store_solve_time, 0,'',showmeans=True,
                          vert=True)
    plt.show()

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task')):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task'))
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task')
    file_path = os.path.join(data_path, 'store_solve_time.npy')
    np.save(file_path, store_solve_time)


def main():
    single_tracking_task()


if __name__ == '__main__':
    main()
