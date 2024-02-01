import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from controller.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType, ControllerType, LiniearizationType
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
    def __init__(self):
        self.system_init()
        self.controller_init()


    def system_init(self):
        l = np.random.uniform(0.2, 0.3)
        r = np.random.uniform(0.02, 0.04)
        # l = 0.23
        # r = 0.036
        self.dt = 0.02
        self.v = 0.2
        self.w = 0.3
        self.twist = np.array([self.v, 0, self.w])
        adj = -SE2Tangent(self.twist).smallAdj()
        self.A = np.eye(3) + self.dt * adj
        self.B = self.dt * np.array([[r / 2, r / 2],
                                     [0, 0],
                                     [-r / l, r / l]])
        self.c = -self.dt * self.twist


        self.Q = 200 * np.eye(3)
        self.R = 0.2 * np.eye(2)


    def controller_init(self):
        self.K0 = -ct.dlqr(self.A, self.B, self.Q, self.R)[0]
        self.K_ini = self.K0
        self.k0 = -np.linalg.pinv(self.B) @ self.c


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

    def solve_with_dp(self):
        # argumented A, B
        n = self.A.shape[0]
        m = self.B.shape[1]
        A = np.zeros((n + n, n + n))
        A[:n, :n] = self.A
        A[:n, n:] = np.eye(n)
        A[n:, n:] = np.eye(n)
        B = np.zeros((n + n, m))
        B[:n, :] = self.B

        # set Q
        Q = np.zeros((n + n, n + n))
        Q[:n, :n] = self.Q

        # set R
        R = self.R

        # backward dp
        S = np.zeros((n + n, n + n))
        S_next = Q
        K = np.zeros((m, n + n))
        for i in range(100000):
            K = -np.linalg.inv(R + B.T @ S_next @ B) @ B.T @ S_next @ A
            S = A.T @ S_next @ A + Q + A.T @ S_next @ B @ K
            S_next = S
        print("K: ", K)

        K_12 = K[:, n:]
        print("res: ", self.B @ K_12)



    def check_feedback_stabilizable(self, K):
        assert K.shape[0] == self.B.shape[1] and K.shape[1] == self.A.shape[0]
        A = self.A + self.B @ K
        eig, _ = np.linalg.eig(A)
        res = True
        for e in eig:
            if np.linalg.norm(e) >= 1:
                res = False
                break
        return res

    def update_K0(self, K):
        self.K0 = K

    def update_k0(self, k):
        self.k0 = k

    def update_B(self, B):
        self.B = B

def simulation(lti, learning=False):
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
    nTraj = 1200
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.zeros((3,)),
                             'linear_vel': lti.twist[0],
                             'angular_vel': lti.twist[2],
                             'nTraj': nTraj,
                             'dt': lti.dt}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    robot = WMRSimulator()
    ref_robot = WMRSimulator()
    if ~learning:
        robot.set_init_state(np.array([-0., -0., 0]))
        ref_robot.set_init_state(np.array([0, 0, 0]))
    state_container = np.zeros((n, nTraj-1))
    x_container = np.zeros((n, nTraj-1))
    ref_state_container = np.zeros((n, nTraj-1))
    for i in range(nTraj-1):
        curr_state = robot.get_state()
        state_container[:, i] = curr_state
        curr_ref = ref_robot.get_state()
        ref_state_container[:, i] = curr_ref
        x = SE2(curr_ref[0], curr_ref[1], curr_ref[2]).between(
            SE2(curr_state[0], curr_state[1], curr_state[2])).log().coeffs()
        x_container[:, i] = x
        u = K @ x + k
        if learning:
            # u = u + np.random.normal(0, 1, (m,))
            u = u + np.random.uniform(-5, 5, (m,))
        next_state, _, _, _, _ = robot.step(u)
        ref_twist = np.array([ref_control[0, i], 0, ref_control[1, i]])
        ref_u = ref_robot.twist_to_control(ref_twist)
        next_ref, _, _, _, _ = ref_robot.step(ref_u)
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

        # plt x
        plt.figure()
        plt.plot(x_container[:3, :].T)
        plt.title('learning: {}'.format(learning))
        legend = ['x', 'y', 'theta']
        plt.legend(legend)

        plt.show()
        print("x: ", x_container[:, -1])

    return data


def B_Indentifier(lti):
    # lti = LTI()
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    iteration = 0
    recovered_B_vector = np.zeros(0, dtype=np.ndarray)
    data_container = simulation(lti, True)
    K_prev = np.zeros((m, n))
    k_prev = np.zeros((m,))
    while np.linalg.norm(data_container.K - K_prev) > 0.05 or np.linalg.norm(data_container.k - k_prev) > 0.05:
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
        recovered_B_vector = np.append(recovered_B_vector, B)
        print("B = ", B)
        k = -np.linalg.pinv(B) @ lti.c
        data_container.K = K
        data_container.k = k
        print("current K: ", K)
        print("current k: ", k)
        print("optimal K: ", lti.get_optimal_K())
        print("error K: ", data_container.K - K_prev)
        print("error k: ", data_container.k - k_prev)
        print("====================================")
        iteration += 1
    print("K: ", K)
    print("k: ", k)
    # print("Recovered B: ", B)
    lti.update_k0(k)
    lti.update_K0(K)
    lti.update_B(B)




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


if __name__ == '__main__':
    lti = LTI()
    simulation(lti, False)
    for i in range(5):
        B_Indentifier(lti)
    simulation(lti, False)
    # learning()
    print("optimal B: ", lti.B)
    print("optimal K = ", lti.get_optimal_K())
    print("optimal k = ", lti.get_optimal_k())




