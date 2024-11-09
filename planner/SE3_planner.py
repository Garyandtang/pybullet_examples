import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from manifpy import SE3, SE3Tangent, SO3, SO3Tangent
import casadi as ca
import math
from utils.enum_class import TrajType

class SE3Planner:
    def __init__(self, config):
        self.nState = 7
        self.nTwist = 6
        if not config:
            config = {'type': TrajType.CONSTANT,
                      'param': {'start_state': np.array([0, 0, 0, 0, 0, 0, 1]),
                                'linear_vel': np.array([0.5, 0, 0]),
                                'angular_vel': np.array([0, 0, 0.5]),
                                'dt': 0.02,
                                'nTraj': 170}}


        if config['type'] == TrajType.CONSTANT:
            self.generate_constant_traj(config['param'])

    def generate_constant_traj(self, config):
        self.dt = config['dt']
        self.nTraj = config['nTraj']
        self.ref_state = np.zeros((self.nState, self.nTraj))
        self.ref_twist = np.zeros((self.nTwist, self.nTraj))
        self.ref_state[:, 0] = config['start_state']
        twist = np.hstack((config['angular_vel'], config['linear_vel']))
        for i in range(self.nTraj-1):
            self.ref_twist[:, i] = twist
            xi = SE3Tangent(twist * self.dt)
            curr_X = SE3(self.ref_state[0:3, i], self.ref_state[3:3+4, i])
            next_X = curr_X + xi
            self.ref_state[:, i+1] = np.hstack((next_X.translation(), next_X.quat().flatten()))

        return self.ref_state, self.ref_twist, self.dt

    def get_traj(self):
        return self.ref_state, self.ref_twist, self.dt



if __name__ == '__main__':
    planner = SE3Planner(None)
    ref_SE3, ref_twist, dt = planner.generate_constant_traj({'start_state': np.array([0, 0, 0, 0, 0, 0, 1]),
                                'linear_vel': np.array([0, 0.5, 0]),
                                'angular_vel': np.array([0, 0, 0.5]),
                                'dt': 0.02,
                                'nTraj': 2500})
    plt.figure()
    plt.plot(ref_SE3[0, :], ref_SE3[2, :], 'r')
    plt.legend(['x', 'y', 'z'])
    plt.show()



