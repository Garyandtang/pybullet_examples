import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from utils.enum_class import WMRType, TrajType
from ref_traj_generator import TrajGenerator
from feedback_linearization import FBLinearizationController
from naive_mpc import NaiveMPC
from error_dynamics_mpc import ErrorDynamicsMPC
from utils.enum_class import WMRType, TrajType, CostType, DynamicsType


def mpc_simulation(traj_generator, controller, init_state):
    init_SE2 = SE2(init_state[0], init_state[1], init_state[2])
    ref_SE2, ref_twist, dt = traj_generator.get_traj()

    # container for recording SE2 state and twist
    nSE2 = 4
    nTwist = 3
    nTraj = ref_SE2.shape[1]
    store_SE2 = np.zeros((nSE2, nTraj))
    store_twist = np.zeros((nTwist, nTraj))
    store_SE2[:, 0] = init_SE2.coeffs()

    t = 0
    for i in range(nTraj - 1):
        if (i == 11):
            print('debug')
        curr_SE2 = store_SE2[:, i]
        curr_vel_cmd = controller.solve(curr_SE2, t)
        curr_twist = controller.vel_cmd_to_local_twist(curr_vel_cmd).full().flatten()
        curr_X = SE2(curr_SE2)
        next_X = curr_X + SE2Tangent(curr_twist) * dt
        store_SE2[:, i + 1] = next_X.coeffs()
        store_twist[:, i] = curr_twist
        t += dt

    return store_SE2, store_twist


def nmpc_simulation(traj_generator, controller, init_state):
    init_SE2 = SE2(init_state[0], init_state[1], init_state[2])
    ref_SE2, ref_twist, dt = traj_generator.get_traj()

    # container for recording SE2 state and twist
    nSE2 = 4
    nTwist = 3
    nTraj = ref_SE2.shape[1]
    store_SE2 = np.zeros((nSE2, nTraj))
    store_twist = np.zeros((nTwist, nTraj))
    store_SE2[:, 0] = init_SE2.coeffs()

    t = 0
    for i in range(nTraj - 1):
        curr_SE2 = store_SE2[:, i]
        curr_X = SE2(curr_SE2)
        curr_state = np.array([curr_X.x(), curr_X.y(), curr_X.angle()])
        curr_vel_cmd = controller.solve(curr_state, t)
        curr_twist = controller.vel_cmd_to_local_twist(curr_vel_cmd).full().flatten()

        next_X = curr_X + SE2Tangent(curr_twist) * dt
        store_SE2[:, i + 1] = next_X.coeffs()
        store_twist[:, i] = curr_twist
        t += dt

    return store_SE2, store_twist


def feedback_linearization_simulation(traj_generator, controller, init_state):
    init_SE2 = SE2(init_state[0], init_state[1], init_state[2])
    ref_SE2, ref_twist, dt = traj_generator.get_traj()

    # container for recording SE2 state and twist
    nSE2 = 4
    nTwist = 3
    nTraj = ref_SE2.shape[1]
    store_SE2 = np.zeros((nSE2, nTraj))
    store_twist = np.zeros((nTwist, nTraj))
    store_SE2[:, 0] = init_SE2.coeffs()

    t = 0
    for i in range(nTraj - 1):
        curr_SE2 = store_SE2[:, i]
        curr_X = SE2(curr_SE2)
        ref_X = SE2(ref_SE2[:, i])
        ref_state = np.array([ref_X.x(), ref_X.y(), ref_X.angle()])
        curr_ref_twist = ref_twist[:, i]
        curr_ref_vel_cmd = np.array([curr_ref_twist[0],curr_ref_twist[2]])
        curr_state = np.array([curr_X.x(), curr_X.y(), curr_X.angle()])
        curr_vel_cmd = controller.feedback_control(curr_state, ref_state, curr_ref_vel_cmd)
        curr_twist = np.array([curr_vel_cmd[0], 0, curr_vel_cmd[1]])

        next_X = curr_X + SE2Tangent(curr_twist) * dt
        store_SE2[:, i + 1] = next_X.coeffs()
        store_twist[:, i] = curr_twist
        t += dt

    return store_SE2, store_twist

def main():
    # set up init state and reference trajectory
    init_state = np.array([-0.5, -0.5, np.pi / 6])
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.4,
                             'angular_vel': 0.5,
                             'nTraj': 700,
                             'dt': 0.02}}

    init_state = np.array([0.2, 0.3, np.pi/4])
    traj_config = {'type': TrajType.TIME_VARYING,
              'param': {'start_state': np.array([0, 0, 0]),
                        'dt': 0.02,
                        'nTraj': 300}}

    # figure of eight
    traj_config = {'type': TrajType.EIGHT,
              'param': {'start_state': np.array([0, 0, 0]),
                        'dt': 0.05,
                        'scale': 1,
                        'nTraj': 300}}
    traj_generator = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()

    model_config = {'cost_type': CostType.POSITION,
                    'dynamics_type': DynamicsType.EULER_FIRST_ORDER}

    fb_linearization_controller = FBLinearizationController()
    edmpc = ErrorDynamicsMPC(traj_config)

    nmpc = NaiveMPC(traj_config, model_config)

    v_max = 1
    w_max = 4
    edmpc.set_control_bound(-v_max,v_max,-w_max,w_max)
    nmpc.set_control_bound(-v_max,v_max,-w_max,w_max)

    SE2_store_ed, twist_store_ed = mpc_simulation(traj_generator, edmpc, init_state)
    SE2_store_nmpc, twist_store_nmpc = nmpc_simulation(traj_generator, nmpc, init_state)
    SE2_store_fb, twist_store_fb = feedback_linearization_simulation(traj_generator, fb_linearization_controller, init_state)

    # plot the result
    plt.figure()
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'b', label='reference')
    plt.plot(SE2_store_ed[0, :], SE2_store_ed[1, :], 'r', label='edmpc')
    plt.plot(SE2_store_nmpc[0, :], SE2_store_nmpc[1, :], 'g', label='nmpc')
    plt.plot(SE2_store_fb[0, :], SE2_store_fb[1, :], 'k', label='fb')
    plt.legend()
    plt.show()

    # plot the error
    error = np.linalg.norm(SE2_store_ed[0:2, :] - ref_SE2[0:2, :], axis=0)
    error_nmpc = np.linalg.norm(SE2_store_nmpc[0:2, :] - ref_SE2[0:2, :], axis=0)
    error_fb = np.linalg.norm(SE2_store_fb[0:2, :] - ref_SE2[0:2, :], axis=0)
    plt.figure()
    plt.plot(error, label='edmpc')
    plt.plot(error_nmpc, label='nmpc')
    plt.plot(error_fb, label='fb')
    plt.legend()
    plt.title('position error')
    plt.show()

    # plot orientation error
    orientation_error_ed = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        curr_ref_SE2 = SE2(ref_SE2[:, i])
        curr_SE2 = SE2(SE2_store_ed[:, i])
        orientation_error = SO2(curr_ref_SE2.angle()).between(SO2(curr_SE2.angle())).log().coeffs()
        orientation_error_ed[i] = scipy.linalg.norm(orientation_error)

    orientation_error_nmpc = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        curr_ref_SE2 = SE2(ref_SE2[:, i])
        curr_SE2 = SE2(SE2_store_nmpc[:, i])
        orientation_error = SO2(curr_ref_SE2.angle()).between(SO2(curr_SE2.angle())).log().coeffs()
        orientation_error_nmpc[i] = scipy.linalg.norm(orientation_error)

    orientation_error_fb = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        curr_ref_SE2 = SE2(ref_SE2[:, i])
        curr_SE2 = SE2(SE2_store_fb[:, i])
        orientation_error = SO2(curr_ref_SE2.angle()).between(SO2(curr_SE2.angle())).log().coeffs()
        orientation_error_fb[i] = scipy.linalg.norm(orientation_error)

    plt.figure()
    plt.plot(orientation_error_ed, label='edmpc')
    plt.plot(orientation_error_nmpc, label='nmpc')
    plt.plot(orientation_error_fb, label='fb')
    plt.legend()
    plt.title('orientation error')
    plt.show()

    # plot the twist
    plt.figure()
    plt.plot(twist_store_ed[0, :], label='edmpc')
    plt.plot(twist_store_nmpc[0, :], label='nmpc')
    plt.plot(twist_store_fb[0, :], label='fb')
    plt.legend()
    plt.title('linear velocity')
    plt.show()

    plt.figure()
    plt.plot(twist_store_ed[2, :], label='edmpc')
    plt.plot(twist_store_nmpc[2, :], label='nmpc')
    plt.plot(twist_store_fb[2, :], label='fb')
    plt.legend()
    plt.title('angular velocity')
    plt.show()
    print('done')


if __name__ == '__main__':
    main()
