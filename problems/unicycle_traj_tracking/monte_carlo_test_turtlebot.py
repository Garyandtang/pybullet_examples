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
import matplotlib.pyplot as plt
import scipy

def main():
    mc_num = 1
    # set init state

    # set trajetory
    traj_config = {'type': TrajType.CIRCLE,
                     'param': {'start_state': np.array([0, 0, 0]),
                              'linear_vel': 0.5,
                              'angular_vel': 0.5,
                              'nTraj': 2500,
                              'dt': 0.02}}
    traj_gen = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = traj_gen.get_traj()
    # store error
    edmpc_position_error = np.zeros((mc_num, ref_SE2.shape[1] ))
    edmpc_orientation_error = np.zeros((mc_num, ref_SE2.shape[1]))

    nmpc_position_error = np.zeros((mc_num, ref_SE2.shape[1] ))
    nmpc_orientation_error = np.zeros((mc_num, ref_SE2.shape[1] ))

    fb_position_error = np.zeros((mc_num, ref_SE2.shape[1] ))
    fb_orientation_error = np.zeros((mc_num, ref_SE2.shape[1]))

    for i in range(mc_num):
        # random init state
        init_x = np.random.uniform(-0.1, 0.1)
        init_y = np.random.uniform(-0.1, 0.1)
        init_theta = np.random.uniform(-np.pi / 6, np.pi / 6)
        init_state = np.array([init_x, init_y, init_theta])
        print('mc_num: ', i)
        controller = ErrorDynamicsMPC(traj_config)
        store_SE2, store_twist = constant_vel_simulation(init_state, controller, traj_gen)
        # calculate position and orientation error
        edmpc_position_error[i, :] = np.linalg.norm(store_SE2[:2, :] - ref_SE2[:2, :], axis=0)
        for j in range(ref_SE2.shape[1]):
            ref_angle = SE2(ref_SE2[:, j]).angle()
            so2_error = SO2(store_SE2[2, j]).between(SO2(ref_angle)).log().coeffs()
            edmpc_orientation_error[i, j] = scipy.linalg.norm(so2_error[0])

        controller = NaiveMPC(traj_config)
        store_SE2, store_twist = constant_vel_simulation(init_state, controller, traj_gen)
        # calculate position and orientation error
        nmpc_position_error[i, :] = np.linalg.norm(store_SE2[:2, :] - ref_SE2[:2, :], axis=0)
        for j in range(ref_SE2.shape[1]):
            ref_angle = SE2(ref_SE2[:, j]).angle()
            so2_error = SO2(store_SE2[2, j]).between(SO2(ref_angle)).log().coeffs()
            nmpc_orientation_error[i, j] = scipy.linalg.norm(so2_error[0])

        controller = FBLinearizationController()
        store_SE2, store_twist = constant_vel_simulation(init_state, controller, traj_gen)
        # calculate position and orientation error
        fb_position_error[i, :] = np.linalg.norm(store_SE2[:2, :] - ref_SE2[:2, :], axis=0)
        for j in range(ref_SE2.shape[1]):
            ref_angle = SE2(ref_SE2[:, j]).angle()
            so2_error = SO2(store_SE2[2, j]).between(SO2(ref_angle)).log().coeffs()
            fb_orientation_error[i, j] = scipy.linalg.norm(so2_error[0])
        # plot
        plt.figure()
        plt.plot(edmpc_position_error.T, label='edmpc')
        plt.title("edmpc position error")
        plt.xlabel("N")
        plt.ylabel("position error")
        plt.show()

        plt.figure()
        plt.plot(edmpc_orientation_error.T, label='edmpc')
        plt.title("edmpc orientation error")
        plt.xlabel("N")
        plt.ylabel("orientation error")
        plt.show()

        plt.figure()
        plt.plot(nmpc_position_error.T, label='nmpc')
        plt.title("nmpc position error")
        plt.xlabel("N")
        plt.ylabel("position error")
        plt.show()

        plt.figure()
        plt.plot(nmpc_orientation_error.T, label='nmpc')
        plt.title("nmpc orientation error")
        plt.xlabel("N")
        plt.ylabel("orientation error")
        plt.show()

        plt.figure()
        plt.plot(fb_position_error.T, label='fb')
        plt.title("fb position error")
        plt.xlabel("N")
        plt.ylabel("position error")
        plt.show()

        plt.figure()
        plt.plot(fb_orientation_error.T, label='fb')
        plt.title("fb orientation error")
        plt.xlabel("N")
        plt.ylabel("orientation error")
        plt.show()

    print('end')
    np.save('data/edmpc_position_error.npy', edmpc_position_error)
    np.save('data/edmpc_orientation_error.npy', edmpc_orientation_error)
    np.save('data/nmpc_position_error.npy', nmpc_position_error)
    np.save('data/nmpc_orientation_error.npy', nmpc_orientation_error)
    np.save('data/fb_position_error.npy', fb_position_error)
    np.save('data/fb_orientation_error.npy', fb_orientation_error)

def constant_vel_simulation(init_state, controller, traj_gen):

    # set env and traj
    env = Turtlebot(gui=False, debug=True, init_state=init_state)
    v_min, v_max, w_min, w_max = env.get_vel_cmd_limit()
    ref_SE2, ref_twist, dt = traj_gen.get_traj()

    # set controller limits
    controller.set_control_bound(v_min, v_max, w_min, w_max)

    # store simulation traj
    nTraj = ref_SE2.shape[1]
    store_state = np.zeros((3, nTraj))
    store_twist = np.zeros((2, nTraj))
    env.draw_ref_traj(ref_SE2)
    t = 0
    for i in range(nTraj - 1):
        curr_state = env.get_state()
        store_state[:, i] = curr_state
        store_twist[:, i] = env.get_twist()
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

    store_state[:, -1] = env.get_state()
    store_twist[:, -1] = env.get_twist()

    return store_state, store_twist


if __name__ == '__main__':
    main()