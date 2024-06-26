import numpy as np
from utils.enum_class import TrajType, ControllerType, EnvType
from controller.nonlinear_mpc import NonlinearMPC
from controller.feedback_linearization import FBLinearizationController
from controller.geometric_mpc import GeometricMPC
from planner.ref_traj_generator import TrajGenerator
from monte_carlo_test_turtlebot import simulation
import os
import matplotlib.pyplot as plt
from utils.enum_class import CostType, DynamicsType, ControllerType

def single_tracking_task():
    # set wheel mobile robot
    env_type = EnvType.SCOUT_MINI
    # set controller
    controller_type = ControllerType.GMPC
    # set init state
    init_state = np.array([0, 0, 0])
    # set reference trajetory
    # traj_config = {'type': TrajType.CIRCLE,
    #                'param': {'start_state': np.array([0, 0, 0]),
    #                          'linear_vel': 0.02,
    #                          'angular_vel': 0.05,
    #                          'nTraj': 1500,
    #                          'dt': 0.02}}
    traj_config = {'type': TrajType.EIGHT,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'dt': 0.02,
                             'v_scale': 1,
                             'nTraj': 2500}}
    traj_gen = TrajGenerator(traj_config)

    if controller_type == ControllerType.NMPC:
        config = {'cost_type': CostType.POSITION_EULER, 'dynamics_type': DynamicsType.EULER_FIRST_ORDER}
        controller = NonlinearMPC(traj_config,config)
    elif controller_type == ControllerType.GMPC:
        controller = GeometricMPC(traj_config)
    elif controller_type == ControllerType.FEEDBACK_LINEARIZATION:
        controller = FBLinearizationController()

    # set control parameters: Q, R, N
    Q = np.array([20000, 20000, 2000])
    R = 0.3
    N = 10
    controller.setup_solver(Q, R, N)

    store_SE2, store_twist, store_solve_time = simulation(init_state, controller, traj_gen, env_type, gui=True)


    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task')):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task'))
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'single_tracking_task')
    file_path = os.path.join(data_path, controller_type.value + '_' + 'store_solve_time.npy')
    np.save(file_path, store_solve_time)


def main():
    single_tracking_task()


if __name__ == '__main__':
    main()
