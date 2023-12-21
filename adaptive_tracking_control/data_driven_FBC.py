import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from controller.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType, ControllerType, LiniearizationType
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
import control as ct

class data_pair:
    def __init__(self):
        self.x = np.zeros((3, 1))
        self.u = np.zeros((2, 1))
        self.x_next = np.zeros((3, 1))
class training_data:
    def __init__(self, n, m):
        self.data_container = np.zeros(0, dtype=data_pair)
        self.K = np.zeros((m, n))
        self.k = np.zeros((n, 1))
        self.c = np.zeros((n, 1))


# x_next = A @ x + B @ u + c
class DataDrivenFBC:
    def __init__(self):
        self.Q = None
        self.R = None
        self.K = None
        self.A = None
        self.B = None
        self.c = None
        self.n = None


        # training parameters
        self.dt = 0.02
        self.twist = np.array([0.5, 0, 0.5])
        self.length = 0.23
        self.radius = 0.036
        self.x0 = np.array([12, 3, -3])

        self.K0 = None
        self.k0 = None

    def action(self, x):
        return self.K0 @ x + self.k0

    def update_K(self, K):
        self.K0 = K

    def update_k(self, k):
        self.k0 = k

    def update_B(self, B):
        self.B = B

    def training_setting(self):
        # set A, B
        self.n = 3
        n = self.n
        twist = self.twist
        adj = -SE2Tangent(twist).smallAdj()
        A = np.zeros((n, n))
        A[:n, :n] = np.eye(n) + adj * self.dt


        print(A)

        # set B
        B = np.zeros((n, 2))
        B[0, 0] = self.dt * self.radius / 2
        B[0, 1] = self.dt * self.radius / 2
        B[2, 0] = -self.dt * self.radius / self.length
        B[2, 1] = self.dt * self.radius / self.length
        print(B)

        # set Q
        Q = np.zeros((n, n))
        Q[:3, :3] = np.eye(3) * 1000 * self.dt
        print(Q)

        # set R
        R = np.eye(2) * 10 * self.dt
        print(R)
        # set c
        self.c = self.twist

        # set K0, k0
        K, _, _ = ct.dlqr(A, B, Q, R)
        self.K0 = -K
        B_inv = np.linalg.pinv(B)
        self.k0 = -B_inv @ self.c

        # print("optimal K: ", K)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def learning(self):
        iteration = 10
        robot = Turtlebot()

        print("init K: ", self.K0)
        print("init k: ", self.k0)
        n = self.A.shape[0]
        m = self.B.shape[1]
        Q = self.Q
        R = self.R

        K_contrainer = self.K0.reshape((n * m))
        k_contrainer = self.k0.reshape((m, 1))
        error_container = np.zeros((iteration, 1))
        for i in range(iteration - 1):
            data_container = self.simulation()
            K_prev = np.zeros((n * m, 1))
            while np.linalg.norm(self.K0.reshape(n * m, 1) - K_prev) > 0.01:
                print("error: ", np.linalg.norm(self.K0.reshape(n * m, 1) - K_prev))
                K_prev = self.K0.reshape(n * m, 1)

                S = self.solve_S_from_data_collect(data_container)

                # solve K, k, B from S
                S_22 = S[n:, n:]
                S_12 = S[:n, n:]
                K = -np.linalg.inv(S_22) @ S_12.T
                self.update_K(K)
                print("K = ", K)
                S_11 = S[:n, :n]
                B = self.A @ np.linalg.inv(S_11 - self.Q) @ S_12
                self.update_B(B)
                B_inv = np.linalg.pinv(B)
                k = -B_inv @ self.c
                self.update_k(k)
                K_contrainer = np.vstack((K_contrainer, K.reshape((n * m, 1))[:, 0]))
                # k_contrainer = np.vstack((k_contrainer, k.reshape((m, 1))[:, 0]))
                print("k = ", k)


    def simulation(self):
        K = self.K0
        k = self.k0
        data = training_data(self.A.shape[0], self.B.shape[1])
        data.K = K
        data.k = k
        robot = Turtlebot()
        n = self.A.shape[0]
        m = self.B.shape[1]
        # generate reference trajectory
        nTraj = 100
        start_state = np.random.uniform(-0.1, 0.1, (3,))
        start_state[2] = 0
        traj_config = {'type': TrajType.CIRCLE,
                        'param': {'start_state': start_state,
                                    'linear_vel': self.twist[0],
                                    'angular_vel': self.twist[2],
                                    'nTraj': nTraj,
                                    'dt': self.dt}}
        traj_gen = TrajGenerator(traj_config)
        ref_state, ref_control, dt = traj_gen.get_traj()
        for i in range(nTraj):
            curr_state = robot.get_state()
            curr_ref = ref_state[:, i]
            x = SE2(curr_ref[0], curr_ref[1], curr_ref[2]).between(SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
            u = self.action(x) + np.random.uniform(-0.2, 0.2, (m,))
            x_next, _, _, _, _ = robot.step(u)
            pair = data_pair()
            pair.x = x.reshape((n,))
            pair.u = u
            pair.x_next = x_next.reshape((n,))
            data.data_container = np.append(data.data_container, pair)

        return data

    def solve_S_from_data_collect(self, data):
        # S can be solved with least square method
        n = data.K.shape[1]
        m = data.K.shape[0]
        K0 = data.K
        k = data.k
        xi = np.zeros((n + m,))  # xi = [x; u]
        zeta = np.zeros((n + m,))  # zeta = [x_next; u_next]
        temp = np.kron(xi.T, xi.T)
        A = np.zeros((data.data_container.shape[0], temp.shape[0]))
        b = np.zeros((data.data_container.shape[0],))
        for i in range(data.data_container.shape[0]):
            x = data.data_container[i].x
            u = data.data_container[i].u
            xi[:n] = x
            xi[n:] = u
            zeta[:n] = data.data_container[i].x_next
            u_next = self.action(x)
            zeta[n:] = u_next
            temp = np.kron(xi.T, xi.T) - np.kron(zeta.T, zeta.T)
            A[i, :] = temp
            b[i] = x.T @ self.Q @ x + u.T @ self.R @ u
        S = np.linalg.lstsq(A, b, rcond=None)[0]
        S = S.reshape((n + m, n + m))
        return S

    # def backward_dp(self):
    #     # backward dp
    #     n = self.A.shape[0]
    #     m = self.B.shape[1]
    #     S = np.zeros((n, n))
    #     S_next = self.Q
    #     K = np.zeros((m, n))
    #     B_inv = np.array([[1388.888888, 0, -159.722222],
    #                       [1388.888888, 0, 159.722222]])
    #     for i in range(1500):
    #         K = -np.linalg.inv(self.R + self.B.T @ S_next @ self.B) @ self.B.T @ S_next @ self.A
    #         K[:, 3:] = -B_inv
    #         S = self.A.T @ S_next @ self.A + self.Q + self.A.T @ S_next @ self.B @ K
    #         S_next = S
    #     self.K = K
    #     print("K: ", K)
    #     print("S: ", S)
    #
    #     # self.K[:, 3:] = -B_inv
    #     print("K: ", self.K)
    #     return K, S
    #
    # def forward_pass(self):
    #     x = np.hstack((self.x0, self.xi))
    #     # x = self.x0
    #     N = 1500
    #     x_traj = np.zeros((N, x.shape[0]))
    #     for i in range(N):
    #         x_traj[i, :] = x.T
    #         u = self.K @ x
    #         x = self.A @ x + self.B @ u
    #     print(x_traj)
    #     plt.figure()
    #     plt.plot(x_traj[:, 0])
    #     plt.plot(x_traj[:, 1])
    #     plt.plot(x_traj[:, 2])
    #     plt.legend(['x', 'y', 'z'])
    #     plt.show()
    #
    #     plt.figure()
    #     plt.plot(x_traj[:, 3])
    #     plt.plot(x_traj[:, 4])
    #     plt.plot(x_traj[:, 5])
    #     plt.legend(['x_dot', 'y_dot', 'z_dot'])
    #     plt.show()
    #
    #     x_end = x_traj[-1, :3]
    #     print("x_end: ", x_end)
    #     XXX = SE2Tangent(x_end).exp().angle()
    #     print(XXX)

    def inv_B(self):
        B = self.B
        B_inv = np.linalg.pinv(B)
        return B_inv



if __name__ == '__main__':
    a = DataDrivenFBC()
    a.training_setting()
    a.learning()
