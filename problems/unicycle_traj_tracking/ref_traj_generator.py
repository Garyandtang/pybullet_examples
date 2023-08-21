import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from utils.enum_class import TrajType


class TrajGenerator:
    def __init__(self, config):
        self.nSE2 = 4
        self.nTwist = 3
        if not config:
            config = {'type': TrajType.CIRCLE,
                      'param': {'start_state': np.array([0, 0, 0]),
                                'linear_vel': 0.5,
                                'angular_vel': 0.5,
                                'dt': 0.05,
                                'nTraj': 170}}
        if config['type'] == TrajType.CIRCLE:
            self.generate_circle_traj(config['param'])
        elif config['type'] == TrajType.EIGHT:
            self.generate_eight_traj(config['param'])
        elif config['type'] == TrajType.POSE_REGULATION:
            self.generate_pose_regulation_traj(config['param'])
        elif config['type'] == TrajType.TIME_VARYING:
            self.generate_time_vary_traj(config['param'])

    def generate_circle_traj(self, config):
        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_SE2 = np.zeros((self.nSE2, self.nTraj))  # [x, y, cos(theta), sin(theta)]
        self.ref_twist = np.zeros((self.nTwist, self.nTraj))  # [vx, vy, w]
        state = config['start_state']
        self.ref_SE2[:, 0] = SE2(state[0], state[1], state[2]).coeffs()
        vel_cmd = np.array([config['linear_vel'], config['angular_vel']])
        v = self.vel_cmd_to_local_vel(vel_cmd)  # constant velocity
        self.ref_twist[:, 0] = v

        for i in range(self.nTraj - 1):  # 0 to nTraj-2
            SE2_coeffs = self.ref_SE2[:, i]
            X = SE2(SE2_coeffs)  # SE2 state
            X = X + SE2Tangent(v * self.dt)  # X * SE2Tangent(xi * self.dt).exp()
            self.ref_SE2[:, i + 1] = X.coeffs()
            self.ref_twist[:, i + 1] = v

    def generate_pose_regulation_traj(self, config):
        # example of pose regulation config
        # config = {'type': TrajType.POSE_REGULATION,
        #           'param': {'end_state': np.array([0, 0, 0]),
        #                     'dt': 0.05,
        #                     'nTraj': 170}}
        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_SE2 = np.zeros((self.nSE2, self.nTraj))
        end_state = config['end_state']
        end_SE2_coeffs = SE2(end_state[0], end_state[1], end_state[2]).coeffs()
        for i in range(self.nTraj):
            self.ref_SE2[:, i] = end_SE2_coeffs
        self.ref_twist = np.zeros((self.nTwist, self.nTraj))


    def generate_time_vary_traj(self, config):

        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_SE2 = np.zeros((self.nSE2, self.nTraj))  # [x, y, cos(theta), sin(theta)]
        self.ref_twist = np.zeros((self.nTwist, self.nTraj))  # [vx, vy, w]
        state = config['start_state']
        self.ref_SE2[:, 0] = SE2(state[0], state[1], state[2]).coeffs()
        vel_cmd = np.array([np.cos(0), np.sin(0)])
        v = self.vel_cmd_to_local_vel(vel_cmd)  # constant velocity
        self.ref_twist[:, 0] = v

        for i in range(self.nTraj - 1):  # 0 to nTraj-2
            SE2_coeffs = self.ref_SE2[:, i]
            twist = self.ref_twist[:, i]
            X = SE2(SE2_coeffs)  # SE2 state
            X = X + SE2Tangent(twist * self.dt)  # X * SE2Tangent(xi * self.dt).exp()
            vel_cmd = np.array([0.8*np.cos((i+1)*self.dt), np.sin(2*(i+1)*self.dt)*2*np.pi])
            print("vel_cmd: ", vel_cmd)
            print("vel_cmd_to_local_vel: ", self.vel_cmd_to_local_vel(vel_cmd))
            self.ref_SE2[:, i + 1] = X.coeffs()
            self.ref_twist[:, i + 1] = self.vel_cmd_to_local_vel(vel_cmd)

    def generate_eight_traj(self, config):
        raise NotImplementedError

    def get_traj(self):
        return self.ref_SE2, self.ref_twist, self.dt

    def vel_cmd_to_local_vel(self, vel_cmd):
        # non-holonomic constraint
        # vel_cmd: [v, w]
        # return: [v, 0, w]
        return np.array([vel_cmd[0], 0, vel_cmd[1]])


def test_traj_generator():
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([1, 1, np.pi]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 170,
                             'dt': 0.05}}
    traj_generator = TrajGenerator(traj_config)
    ref_traj, ref_v, dt = traj_generator.get_traj()
    plt.figure(1)
    plt.plot(ref_traj[0, :], ref_traj[1, :], 'b')
    plt.title('Reference Trajectory')
    plt.show()

    # convert to [x, y, theta]
    traj = np.zeros((3, ref_traj.shape[1]))
    for i in range(ref_traj.shape[1]):
        X = SE2(ref_traj[:, i])
        traj[:, i] = np.array([X.x(), X.y(), X.angle()])

    plt.figure(2)
    plt.plot(traj[0, :], traj[1, :], 'b')
    plt.title('Reference Trajectory [x, y, theta]')
    plt.show()


def test_pose_regulation_traj_generator():
    config = {'type': TrajType.POSE_REGULATION,
              'param': {'end_state': np.array([0, 0, 0]),
                        'dt': 0.05,
                        'nTraj': 170}}
    traj_generator = TrajGenerator(config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    plt.figure(1)
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'b')
    plt.title('Reference Trajectory')
    plt.show()

def test_time_varying_traj_generator():
    config = {'type': TrajType.TIME_VARYING,
              'param': {'start_state': np.array([0, 0, 0]),
                        'dt': 0.02,
                        'nTraj': 300}}
    traj_generator = TrajGenerator(config)
    ref_SE2, ref_twist, dt = traj_generator.get_traj()
    plt.figure(1)
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'b')
    plt.title('Reference Trajectory')
    plt.show()


if __name__ == '__main__':
    test_time_varying_traj_generator()
