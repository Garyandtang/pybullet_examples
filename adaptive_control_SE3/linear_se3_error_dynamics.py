import control as ct
import numpy as np
from adaptive_control_SE3.lie_utils import *
class LinearSE3ErrorDynamics:
    def __init__(self, fixed_param=False):
        self.system_init(fixed_param)
        self.controller_init()
        # self.get_ground_truth_control()


    def system_init(self, fixed_param=False):
        if fixed_param:
            self.I = np.eye(3)
            self.m = 1.0
        else:
            self.I = np.eye(3) + 0.1 * np.random.randn(3, 3)
            self.m = 1.0 + 0.1 * np.random.randn(1)
        # generalized inertia matrix
        J = np.zeros((6, 6))
        J[0:3, 0:3] = self.I
        J[3:6, 3:6] = self.m * np.eye(3)
        invJ = np.linalg.inv(J)

        self.v = np.array([2, 0, 0.2])
        self.w = np.array([0, 0, 1])
        self.dt = 0.02
        self.twist = np.hstack([self.v, self.w])

        self.A = np.zeros((12, 12))
        Ac = np.zeros((12, 12))
        H = invJ @ smallAdjointInv(self.w, self.v) @ J + invJ @ gamma_right(self.I, self.m, self.w, self.v)
        Ac[0:6, 0:6] = -smallAdjoint(self.w, self.v)
        Ac[0:6, 6:12] = np.eye(6)
        Ac[6:12, 6:12] = H
        self.A = np.eye(12) + Ac * self.dt

        self.B = np.zeros((12, 6))
        self.B[6:12, :] = invJ * self.dt

        self.Q = np.diag(np.array([10, 10, 10, 1, 1, 1, 20, 20, 20, 1, 1, 1])) * 10
        self.R = np.eye(6) * 1e-5




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


    def controller_init(self):
        self.K0 = -ct.dlqr(self.A, self.B, self.Q, self.R)[0]
        self.K_ini = self.K0
        self.B_ini = self.B


if __name__ == '__main__':
    linear_se3_error_dynamics = LinearSE3ErrorDynamics()
    print(linear_se3_error_dynamics.A)
    print(linear_se3_error_dynamics.B)
    print("K: ", linear_se3_error_dynamics.K0)
    print(linear_se3_error_dynamics.I)
    print(linear_se3_error_dynamics.m)
    print(linear_se3_error_dynamics.v)
    print(linear_se3_error_dynamics.w)
    print(linear_se3_error_dynamics.twist)
    print(linear_se3_error_dynamics.Q)
    print(linear_se3_error_dynamics.R)

