import numpy as np
from utils.enum_class import TrajType, ControllerType, EnvType
from controller.naive_mpc import NaiveMPC
from controller.feedback_linearization import FBLinearizationController
from controller.error_dynamics_mpc import ErrorDynamicsMPC
from controller.ref_traj_generator import TrajGenerator
from monte_carlo_test_turtlebot import simulation
import os
import matplotlib.pyplot as plt

def single_tracking_task():
    env_type = EnvType.SCOUT_MINI
    controller_type = ControllerType.GMPC
    init_state = np.array([0, 0, 0])
    traj_config = {'type': TrajType.EIGHT,
              'param': {'start_state': np.array([0, 0, 0]),
                        'dt': 0.02,
                        'scale': 2,
                        'nTraj': 2500}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    if controller_type == ControllerType.NMPC:
        controller = NaiveMPC(traj_config)
    elif controller_type == ControllerType.GMPC:
        controller = ErrorDynamicsMPC(traj_config)
    elif controller_type == ControllerType.FEEDBACK_LINEARIZATION:
        controller = FBLinearizationController()

    store_SE2, store_twist, store_solve_time = simulation(init_state, controller, traj_gen, env_type, gui=True)

    # quartile chart
    plt.figure()
    plt.boxplot(store_solve_time, 0, '')
    plt.show()

    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task')):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task'))
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task')
    file_path = os.path.join(data_path, controller_type.value + '_' + 'store_solve_time.npy')
    np.save(file_path, store_solve_time)


def main():
    single_tracking_task()


if __name__ == '__main__':
    main()
