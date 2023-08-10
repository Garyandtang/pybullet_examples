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

    def generate_circle_traj(self, config):
        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_traj = np.zeros((4, self.nTraj))  # [x, y, cos(theta), sin(theta)]
        self.ref_v = np.zeros((3, self.nTraj))  # [vx, vy, w]
        state = config['start_state']
        self.ref_traj[:, 0] = SE2(state[0], state[1], state[2]).coeffs()
        vel_cmd = np.array([config['linear_vel'], config['angular_vel']])
        v = self.vel_cmd_to_local_vel(vel_cmd)  # constant velocity
        self.ref_v[:, 0] = v

        for i in range(self.nTraj - 1):  # 0 to nTraj-2
            SE2_coeffs = self.ref_traj[:, i]
            X = SE2(SE2_coeffs)  # SE2 state
            X = X + SE2Tangent(v * self.dt)  # X * SE2Tangent(xi * self.dt).exp()
            self.ref_traj[:, i + 1] = X.coeffs()
            self.ref_v[:, i + 1] = v

    def generate_eight_traj(self, config):
        raise NotImplementedError

    def get_traj(self):
        return self.ref_traj, self.ref_v, self.dt

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


if __name__ == '__main__':
    test_traj_generator()
