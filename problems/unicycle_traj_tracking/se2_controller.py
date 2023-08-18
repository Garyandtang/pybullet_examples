import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from utils.enum_class import WMRType, TrajType
from ref_traj_generator import TrajGenerator
class SE2Controller:
    def __init__(self, Kp = np.array([1, 1, 1])):
        self.Kp = Kp

    def feedback_control(self, curr_SE2, ref_SE2):
        """
        :param curr_SE2: [x, y, cos(theta), sin(theta)]
        :param ref_SE2: [x_d, y_d, cos(theta_d), sin(theta_d)]
        :return: local twist:[v_x, v_y, w]
        """
        X = SE2(curr_SE2)
        X_d = SE2(ref_SE2)
        X_err = X.between(X_d)
        se2_err = X_err.log()
        twist = self.Kp * se2_err.coeffs()
        return twist

    def feedback_feedforward_control(self, curr_SE2, ref_SE2, ref_twist):
        twist_fb = self.feedback_control(curr_SE2, ref_SE2)
        twist_ff = ref_twist
        twist = twist_fb + twist_ff
        return twist

    def local_twsit_to_vel_cmd(self, local_twist, type=WMRType.UNICYCLE):
        if type == WMRType.UNICYCLE:
            return np.array([local_twist[0], local_twist[2]])

    def vel_cmd_to_local_twist(self, vel_cmd, type=WMRType.UNICYCLE):
        if type == WMRType.UNICYCLE:
            return np.array([vel_cmd[0], 0, vel_cmd[1]])



def test_se2_controller():
    # set up init state and reference trajectory
    init_state = np.array([-0.2, -0.2, np.pi/6])
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 6000,
                             'dt': 0.02}}
    traj_generator = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    se2_controller = SE2Controller()

    # container for recording SE2 state and twist
    nSE2 = 4
    nTwist = 3
    store_SE2 = np.zeros((nSE2, ref_SE2.shape[1]))
    store_twist = np.zeros((nTwist, ref_SE2.shape[1]))
    store_SE2[:, 0] = SE2(init_state[0], init_state[1], init_state[2]).coeffs()

    t = 0
    for i in range(ref_SE2.shape[1]-1):
        curr_SE2 = store_SE2[:, i]
        curr_ref_SE2 = ref_SE2[:, i]
        curr_ref_twist = ref_twist[:, i]
        curr_twist = se2_controller.feedback_feedforward_control(curr_SE2, curr_ref_SE2, curr_ref_twist)
        curr_vel_cmd = se2_controller.local_twsit_to_vel_cmd(curr_twist)
        curr_twist = se2_controller.vel_cmd_to_local_twist(curr_vel_cmd)
        store_twist[:, i] = curr_twist
        # next SE2 state
        next_SE2 = SE2(curr_SE2) + SE2Tangent(curr_twist) * dt
        store_SE2[:, i + 1] = next_SE2.coeffs()
        t += dt

    # plot
    plt.figure()
    plt.plot(store_SE2[0, :], store_SE2[1, :], 'b')
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    plt.show()

    # plot distance error
    plt.figure()
    plt.plot(np.linalg.norm(store_SE2[0:2, :] - ref_SE2[0:2, :], axis=0))
    plt.title('distance error')
    plt.show()

    # plot angle error
    plt.figure()
    orientation_store = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        X_d = SE2(ref_SE2[:, i])
        X = SE2(store_SE2[:, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())


    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')
    plt.show()


def test_pose_regulation():
    init_state = np.array([-0.01, 0, 0])
    config = {'type': TrajType.POSE_REGULATION,
              'param': {'end_state': np.array([0, 0, np.pi/2]),
                        'dt': 0.05,
                        'nTraj': 1700}}
    traj_generator = TrajGenerator(config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    se2_controller = SE2Controller()
    # container for recording SE2 state and twist
    nSE2 = 4
    nTwist = 3
    store_SE2 = np.zeros((nSE2, ref_SE2.shape[1]))
    store_twist = np.zeros((nTwist, ref_SE2.shape[1]))
    store_SE2[:, 0] = SE2(init_state[0], init_state[1], init_state[2]).coeffs()

    t = 0
    for i in range(ref_SE2.shape[1]-1):
        curr_SE2 = store_SE2[:, i]
        curr_ref_SE2 = ref_SE2[:, i]
        curr_ref_twist = ref_twist[:, i]
        curr_twist = se2_controller.feedback_feedforward_control(curr_SE2, curr_ref_SE2, curr_ref_twist)
        # curr_vel_cmd = se2_controller.local_twsit_to_vel_cmd(curr_twist)
        # curr_twist = se2_controller.vel_cmd_to_local_twist(curr_vel_cmd)
        store_twist[:, i] = curr_twist
        # next SE2 state
        next_SE2 = SE2(curr_SE2) + SE2Tangent(curr_twist) * dt
        store_SE2[:, i + 1] = next_SE2.coeffs()
        t += dt

    # plot
    plt.figure()
    plt.plot(store_SE2[0, :], store_SE2[1, :], 'b')
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    plt.show()

    # # plot (x, y, theta) pose figure with arrow
    # plt.figure()
    # plt.plot(store_SE2[0, :], store_SE2[1, :], 'b')
    # plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    # plt.quiver(store_SE2[0, :], store_SE2[1, :], np.cos(store_SE2[2, :]), np.sin(store_SE2[2, :]), color='b')
    # plt.quiver(ref_SE2[0, :], ref_SE2[1, :], np.cos(ref_SE2[2, :]), np.sin(ref_SE2[2, :]), color='r')
    # plt.show()




    # plot distance error
    plt.figure()
    plt.plot(np.linalg.norm(store_SE2[0:2, :] - ref_SE2[0:2, :], axis=0))
    plt.title('distance error')
    plt.show()

    # plot angle error
    plt.figure()
    orientation_store = np.zeros(ref_SE2.shape[1])
    for i in range(ref_SE2.shape[1]):
        X_d = SE2(ref_SE2[:, i])
        X = SE2(store_SE2[:, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())

    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')
    plt.show()



if __name__ == '__main__':
    test_pose_regulation()



