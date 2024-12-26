import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from utilsStuff.enum_class import TrajType, ControllerType, LiniearizationType
from planner.ref_traj_generator import TrajGenerator

from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from environments.numerical_simulator.WMR_simulator import WMRSimulator

class data_pair:
    def __init__(self, n, m):
        self.x = np.zeros((n,))
        self.u = np.zeros((m, ))
        self.x_next = np.zeros((n,))
class training_data:
    def __init__(self, n, m):
        self.data_container = data_pair(n, m)
        self.K = np.zeros((m, n))
        self.k = np.zeros((n,))
        self.c = np.zeros((n,))
        self.Q = np.zeros((n, n))
        self.R = np.zeros((m, m))
        self.A = np.zeros((n, n))
        self.B = np.zeros((n, m))

class LTI:
    def __init__(self, fixed_param=False):
        self.system_init(fixed_param)
        self.controller_init()
        self.get_ground_truth_control()

    def get_ground_truth_control(self):
        ground_truth_l = 0.23
        ground_truth_r = 0.036
        ground_truth_B = self.dt * np.array([[ground_truth_r / 2, ground_truth_r / 2],
                              [0, 0],
                              [-ground_truth_r / ground_truth_l, ground_truth_r / ground_truth_l]])

        self.k_ground_truth = -np.linalg.pinv(ground_truth_B) @ self.c
        self.K_ground_truth = -ct.dlqr(self.A, ground_truth_B, self.Q, self.R)[0]
        self.B_ground_truth = ground_truth_B


    def system_init(self, fixed_param=False):
        if fixed_param:
            l = 0.15
            r = 0.05
        else:
            l = np.random.uniform(0.15, 0.3)
            r = np.random.uniform(0.03, 0.05)

        self.dt = 0.02
        self.v = 0.02
        self.w = 0.2
        self.twist = np.array([self.v, 0, self.w])
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


    def controller_init(self):
        self.K0 = -ct.dlqr(self.A, self.B, self.Q, self.R)[0]
        self.K_ini = self.K0
        self.k0 = -np.linalg.pinv(self.B) @ self.c
        self.k_ini = self.k0
        self.B_ini = self.B
        self.r_init, self.l_init = calculate_r_l(self.B, self.dt)

    def step(self, x, u):
        # assert u.shape[0] == self.B.shape[1]
        # assert x.shape[0] == self.A.shape[0]
        x_next = self.A @ x + self.B @ u + self.c
        return x_next

    def action(self, x):
        return self.K0 @ x + self.k0

    def cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u

    def get_optimal_K(self):
        K, S, E = ct.dlqr(self.A, self.B, self.Q, self.R)
        return -K

    def get_optimal_k(self):
        k = -np.linalg.pinv(self.B) @ self.c
        return k
    def check_controllable(self):
        C = ct.ctrb(self.A, self.B)
        rank = np.linalg.matrix_rank(C)
        return rank == self.A.shape[0]



    def update_K0(self, K):
        self.K0 = K

    def update_k0(self, k):
        self.k0 = k

    def update_B(self, B):
        self.B = B

def evaluation(lti, nTraj=500, learning=False):
    K = lti.K0
    k = lti.k0
    n = lti.A.shape[0]
    m = lti.B.shape[1]

    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.zeros((3,)),
                             'linear_vel': lti.twist[0],
                             'angular_vel': lti.twist[2],
                             'nTraj': nTraj,
                             'dt': lti.dt}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_velocity, dt = traj_gen.get_traj()
    robot = WMRSimulator()

    state_container = np.zeros((n, nTraj - 1))
    ref_state_container = np.zeros((n, nTraj - 1))
    control_container = np.zeros((m, nTraj - 1))
    ref_control_container = np.zeros((m, nTraj - 1))
    wheel_velocity_container = np.zeros((2, nTraj - 1))
    error_container = np.zeros((n, nTraj - 1))
    for i in range(nTraj - 1):
        curr_state = robot.get_state()
        state_container[:, i] = curr_state
        curr_ref = ref_state[:, i]
        ref_state_container[:, i] = curr_ref
        x = SE2(curr_ref[0], curr_ref[1], curr_ref[2]).between(
            SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
        error_container[:, i] = x
        u = K @ x + k
        if learning:
            # u = u + np.sin(np.random.normal(0, 1, (m,)) * 0.1) + np.cos(np.random.normal(0, 1, (m,)) * 0.1)
            u = u + np.random.normal(0, 0.1, (m,))
            # u = u + np.random.uniform(-0.2, 0.2, (m,))
        control_container[:, i] = u
        # control_container[:, i] = wheel_velocity_container[:, i]
        ref_control_container[:, i] = robot.twist_to_control(np.array([ref_velocity[0, i], 0, ref_velocity[1, i]]))

        next_state, _, _, _, _ = robot.step(u)

    return state_container[0, :], state_container[1, :], error_container, control_container


def simulation(lti, learning=False, variance=0.1):
    K = lti.K0
    k = lti.k0
    print("init K: ", lti.K_ini)
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    data = training_data(n, m)
    data.K = K
    data.k = k
    data.c = lti.c
    data.Q = lti.Q
    data.R = lti.R
    data.A = lti.A
    data.B = lti.B
    nTraj = 1300
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.zeros((3,)),
                             'linear_vel': lti.twist[0],
                             'angular_vel': lti.twist[2],
                             'nTraj': nTraj,
                             'dt': lti.dt}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_velocity, dt = traj_gen.get_traj()
    robot = WMRSimulator()
    # robot = Turtlebot()
    # if ~learning:
    #     robot.set_init_state(np.array([0., 0., 0]))
    error_container = np.zeros((n, nTraj - 1))
    state_container = np.zeros((n, nTraj-1))
    ref_state_container = np.zeros((n, nTraj-1))
    control_container = np.zeros((m, nTraj-1))
    ref_control_container = np.zeros((m, nTraj-1))
    wheel_velocity_container = np.zeros((2, nTraj-1))
    for i in range(nTraj-1):
        curr_state = robot.get_state()
        state_container[:, i] = curr_state
        curr_ref = ref_state[:, i]
        # print("curr_ref: ", curr_ref)
        # print("ref_state: ", ref_state[:, i])
        ref_state_container[:, i] = curr_ref
        x = SE2(curr_ref[0], curr_ref[1], curr_ref[2]).between(
            SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
        error_container[:, i] = x
        u = K @ x + k
        if learning:
            # u = u + np.sin(np.random.normal(0, 1, (m,)) * 0.1) + np.cos(np.random.normal(0, 1, (m,)) * 0.1)
            # u = u + 0.5 * np.random.normal(0, 1, (m,))
            u = u + np.random.normal(0, variance, (m,))
            # u = u + np.random.uniform(-0.2, 0.2, (m,))
        # wheel_velocity_container[:, i] = robot.get_wheel_vel()
        control_container[:, i] = u
        control_container[:, i] = wheel_velocity_container[:, i]
        ref_control_container[:, i] = robot.twist_to_control(np.array([ref_velocity[0, i], 0, ref_velocity[1, i]]))

        next_state, _, _, _, _ = robot.step(u)

        next_ref = ref_state[:, i+1]
        x_next = SE2(next_ref[0], next_ref[1], next_ref[2]).between(
            SE2(next_state[0], next_state[1], next_state[2])).log().coeffs()
        pair = data_pair(n, m)
        pair.x = x
        pair.u = u
        pair.x_next = x_next
        if i == 0:
            data.data_container = pair
        else:
            data.data_container = np.append(data.data_container, pair)

    if not learning:
        # plt ref traj and state traj
        plt.figure()
        plt.plot(ref_state[0, :nTraj - 1], ref_state[1, :nTraj - 1], 'b')
        plt.plot(state_container[0, :nTraj - 1], state_container[1, :nTraj - 1], 'r')
        plt.title('learning: {}'.format(learning))
        legend = ['ref', 'actual']
        plt.legend(legend)
        plt.show()

        # plt error
        plt.figure()
        plt.plot(error_container[:3, :].T)
        plt.title('learning: {}'.format(learning))
        legend = ['x', 'y', 'theta']
        plt.legend(legend)

        plt.show()
        print("x: ", error_container[:, -1])

        # plt control[0] and ref_control[0]
        plt.figure()
        plt.plot(control_container[0, :], 'r')
        plt.plot(ref_control_container[0, :], 'b')
        plt.plot(wheel_velocity_container[0, :], 'g')
        plt.title('learning: {}'.format(learning))
        legend = ['control', 'ref_control', 'wheel_velocity']
        plt.legend(legend)
        plt.show()

        # plt control[1] and ref_control[1]
        plt.figure()
        plt.plot(control_container[1, :], 'r')
        plt.plot(ref_control_container[1, :], 'b')
        plt.plot(wheel_velocity_container[1, :], 'g')
        plt.title('learning: {}'.format(learning))
        legend = ['control', 'ref_control', 'wheel_velocity']
        plt.legend(legend)
        plt.show()



    return data


def projection_B(B, dt):
    r = (B[0, 0] + B[0, 1]) / dt
    l = (r * dt) / (B[2, 1] - B[2, 0]) * 2
    B[0, 0] = r * dt / 2
    B[0, 1] = r * dt / 2
    B[2, 0] = -r * dt / l
    B[2, 1] = r * dt / l
    B[1, 0] = 0
    B[1, 1] = 0
    return B

def calculate_r_l(B, dt):
    r = (B[0, 0] + B[0, 1]) / dt
    l = (r * dt) / (B[2, 1] - B[2, 0]) * 2
    return r, l

def learning(lti, variance=0.1):
    # lti = LTI()
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    iteration = 0
    recovered_B_vector = np.zeros(0, dtype=np.ndarray)
    data_container = simulation(lti, True, variance)
    K_prev = np.zeros((m, n))
    k_prev = np.zeros((m,))
    while np.linalg.norm(data_container.K - K_prev) > 0.01 or np.linalg.norm(data_container.k - k_prev) > 0.01:
        # if iteration > 0:
        #     break
        K_prev = data_container.K
        k_prev = data_container.k
        print("iteration: ", iteration, '===================')
        # solve S from data
        gamma = 1
        S = solve_S_from_data_collect(data_container, data_container.Q, data_container.R, gamma)
        # solve K from S
        S_22 = S[n:n + m, n:n + m]
        S_12 = S[:n, n:n + m]
        K = -np.linalg.inv(S_22) @ S_12.T
        S_11 = S[:n, :n]
        B = lti.A @ np.linalg.inv(S_11 - lti.Q) @ S_12
        B = projection_B(B, lti.dt)
        recovered_B_vector = np.append(recovered_B_vector, B)
        print("B = ", B)
        k = -np.linalg.pinv(B) @ lti.c
        data_container.K = K
        data_container.k = k
        print("current K: ", K)
        print("current k: ", k)
        print("optimal K: ", lti.K_ground_truth)
        print("error K: ", data_container.K - K_prev)
        print("error k: ", data_container.k - k_prev)
        print("====================================")
        iteration += 1
        if iteration > 100:
            return K, k, B, False
    print("K: ", K)
    print("k: ", k)
    # print("Recovered B: ", B)
    lti.update_k0(k)
    lti.update_K0(K)
    lti.update_B(B)
    return K, k, B, True




def solve_S_from_data_collect(data, Q, R, gamma):
    # S can be solved with least square method
    n = data.K.shape[1]
    m = data.K.shape[0]
    K = data.K
    k = data.k
    c = data.c
    xi = np.zeros((n + m + n,))   # xi = [x; u; c]
    zeta = np.zeros((n + m + n,))  # zeta = [x_next; u_next; c]
    temp = np.kron(xi.T, zeta.T)
    A = np.zeros((data.data_container.shape[0], temp.shape[0]))
    b = np.zeros((data.data_container.shape[0],))
    for i in range(data.data_container.shape[0]):
        x = data.data_container[i].x
        u = data.data_container[i].u - k
        x_next = data.data_container[i].x_next
        xi[:n] = x
        xi[n:n+m] = u
        xi[n+m:] = c
        zeta[:n] = x_next
        u_next = K @ x_next
        zeta[n:n+m] = u_next
        zeta[n+m:] = c
        temp = np.kron(xi.T, xi.T) - np.kron(zeta.T, zeta.T)
        A[i, :] = temp
        b[i] = (x.T @ Q @ x + u.T @ R @ u) * np.power(gamma, i)
    S = np.linalg.pinv(A) @ b
    S = S.reshape((2*n + m, 2*n + m))
    return S

# def MonteCarlo():
#     totalSim = 1
#     iteration = 5
#     y_max = 0.8
#     y_min = -0.3
#     K0_container = np.zeros((totalSim, iteration))
#     K1_container = np.zeros((totalSim, iteration))
#     K2_container = np.zeros((totalSim, iteration))
#     K3_container = np.zeros((totalSim, iteration))
#     K4_container = np.zeros((totalSim, iteration))
#     K5_container = np.zeros((totalSim, iteration))
#     k0_container = np.zeros((totalSim, iteration))
#     k1_container = np.zeros((totalSim, iteration))
#     lti = LTI()
#
#     for i in range(totalSim):
#
#         optimal_K = lti.K_optimal
#         optimal_k = lti.k_optimal
#         for j in range(iteration):
#             K, k, B = learning(lti)
#             K0_container[i, j] = K[0, 0] - optimal_K[0, 0]
#             K1_container[i, j] = K[0, 1] - optimal_K[0, 1]
#             K2_container[i, j] = K[0, 2] - optimal_K[0, 2]
#             K3_container[i, j] = K[1, 0] - optimal_K[1, 0]
#             K4_container[i, j] = K[1, 1] - optimal_K[1, 1]
#             K5_container[i, j] = K[1, 2] - optimal_K[1, 2]
#             k0_container[i, j] = k[0] - optimal_k[0]
#             k1_container[i, j] = k[1] - optimal_k[1]
#
#     print("init K: ", lti.K_ini)
#     print("init k: ", lti.k_ini)
#     print("optimal K: ", lti.K_optimal)
#     print("optimal k: ", lti.k_optimal)
#     print("K0: ", lti.K0)
#     print("k0: ", lti.k0)
#     plt.figure()
#     plt.plot(K0_container.T)
#     plt.title("K0")
#     plt.ylim([y_min, y_max])
#     plt.show()
#
#     plt.figure()
#     plt.plot(K1_container.T)
#     plt.title("K1")
#     plt.ylim([y_min, y_max])
#     plt.show()
#
#     plt.figure()
#     plt.plot(K2_container.T)
#     plt.title("K2")
#     plt.ylim([y_min, y_max])
#     plt.show()
#
#     plt.figure()
#     plt.plot(K3_container.T)
#     plt.title("K3")
#     plt.ylim([y_min, y_max])
#     plt.show()
#
#     plt.figure()
#     plt.plot(K4_container.T)
#     plt.title("K4")
#     plt.ylim([y_min, y_max])
#     plt.show()
#
#     plt.figure()
#     plt.plot(K5_container.T)
#     plt.title("K5")
#     plt.ylim([y_min, y_max])
#     plt.show()
#
#     plt.figure()
#     plt.plot(k0_container.T)
#     plt.title("k0")
#     plt.ylim([y_min, y_max])
#     plt.show()
#
#     plt.figure()
#     plt.plot(k1_container.T)
#     plt.title("k1")
#     plt.ylim([y_min, y_max])
#     plt.show()
#     print("end")


if __name__ == '__main__':
    MonteCarlo()




