import os
import numpy as np
from utils.enum_class import TrajType, EnvType, LiniearizationType
from controller.error_dynamics_mpc import ErrorDynamicsMPC
from controller.ref_traj_generator import TrajGenerator
from monte_carlo_test_turtlebot import simulation, calulate_trajecotry_error
from matplotlib import pyplot as plt

def main():
    init_state = np.array([-0.3, -0.3, np.pi/6])
    env_type = EnvType.TURTLEBOT
    root_dir = os.path.join(os.getcwd())
    data_dir = os.path.join(root_dir, 'data', 'linearization_scheme')
    traj_config = {'type': TrajType.CIRCLE,
              'param': {'start_state': np.array([0, 0, 0]),
                        'linear_vel': 0.5,
                        'angular_vel': 0.5,
                        'dt': 0.02,
                        'nTraj': 800}}

    controller = ErrorDynamicsMPC(traj_config)
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()

    store_state, store_control, store_solve_time = simulation(init_state, controller, traj_gen, env_type, gui=False)
    controller = ErrorDynamicsMPC(traj_config, linearization_type=LiniearizationType.WEDGE)
    store_state_1, store_control_1, store_solve_time_1 = simulation(init_state, controller, traj_gen, env_type, gui=False)

    position_error, orientation_error = calulate_trajecotry_error(store_state, ref_state)
    position_error_1, orientation_error_1 = calulate_trajecotry_error(store_state_1, ref_state)

    t = np.arange(0, position_error.shape[0] * 0.02, 0.02)
    font_size = 12
    root_dir = os.path.join(os.getcwd())
    # plot position error
    plt.figure()
    plt.plot(t, position_error[:], label='Linearization Scheme in (17)')
    plt.plot(t, position_error_1[:], label='Linearization Scheme in (16)')
    plt.xlabel("$t$")
    plt.ylabel("$e_p(t)$")
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'position_error.jpg'))
    plt.show()

    # plot orientation error
    plt.figure()
    plt.plot(t, orientation_error[:], label='Linearization Scheme in (17)')
    plt.plot(t, orientation_error_1[:], label='Linearization Scheme in (16)')
    plt.xlabel("$t$")
    plt.ylabel("$e_R(t)$")
    plt.tight_layout()
    plt.legend(fontsize=font_size)
    plt.savefig(os.path.join(data_dir, 'orientation_error.jpg'))
    plt.show()

    # plot trajectory
    plt.figure()
    plt.plot(store_state[0, :], store_state[1,:], label='trajectory with (17)')
    plt.plot(store_state_1[0, :], store_state_1[1,:], label='trajectory with (16)')
    plt.plot(ref_state[0, :], ref_state[1,:], label='reference trajectory')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'trajectory.jpg'))
    plt.show()





if __name__ == '__main__':
    main()