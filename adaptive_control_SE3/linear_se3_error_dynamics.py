import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from utilsStuff.enum_class import TrajType, ControllerType, LiniearizationType
from planner.ref_traj_generator import TrajGenerator
class LinearSE3ErrorDynamics:
    def __init__(self, fixed_param=False):
        self.system_init(fixed_param)
        self.controller_init()
        self.get_ground_truth_control()


    def calculate_K_k(self, A, B):
        K = -ct.dlqr(A, B, self.Q, self.R)[0]
        k = -np.linalg.pinv(B) @ self.c
        return K, k

    def system_init(self, fixed_param=False):
        if fixed_param:
            self.I = np.eye(3)
            self.m = 1.0
        else:
            self.I = np.eye(3) + 0.1 * np.random.randn(3, 3)
            self.m = 1.0 + 0.1 * np.random.randn(1)

        self.v = np.array([2, 0, 0.2])
        self.w = np.array([0, 0, 1])
        self.dt = 0.02
        self.twist = np.hstack([self.v, self.w])

        adj = -SE2Tangent(self.twist).smallAdj()
        self.A = np.eye(3) + self.dt * adj
        self.B = self.dt * np.array([[r / 2, r / 2],
                                     [0, 0],
                                     [-r / l, r / l]])
        self.c = -self.dt * self.twist

        self.init_A = self.A
        self.init_B = self.B
        self.init_c = self.c

        self.Q = 200 * np.eye(3)
        self.R = 1 * np.eye(2)

    def get_ground_truth_control(self):
        ground_truth_l = 0.23
        ground_truth_r = 0.036
        ground_truth_B = self.dt * np.array([[ground_truth_r / 2, ground_truth_r / 2],
                              [0, 0],
                              [-ground_truth_r / ground_truth_l, ground_truth_r / ground_truth_l]])

        self.k_ground_truth = -np.linalg.pinv(ground_truth_B) @ self.c
        self.K_ground_truth = -ct.dlqr(self.A, ground_truth_B, self.Q, self.R)[0]
        self.B_ground_truth = ground_truth_B
        self.A_ground_truth = self.A
