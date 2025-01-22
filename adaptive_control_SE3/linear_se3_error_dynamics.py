import control as ct
import numpy as np
from adaptive_control_SE3.lie_utils import *
from planner.SE3_planner import SE3Planner
from utilsStuff.enum_class import TrajType
from environments.numerical_simulator.single_rigid_body_simulator import SingleRigidBodySimulator
from manifpy import SE3, SO3, SE3Tangent, SO3Tangent
import matplotlib.pyplot as plt
class LinearSE3ErrorDynamics:
    def __init__(self, fixed_param=True):
        self.v = np.array([2, 0, 0.2])
        self.w = np.array([0, 0, 1])
        self.system_init(fixed_param)
        self.controller_init()
        # self.get_ground_truth_control()


    def system_init(self, fixed_param=True):
        if fixed_param:
            self.I = np.eye(3)
            self.m = 1
        else:
            self.I = np.eye(3)
            self.m = 1.0
        # generalized inertia matrix
        J = np.zeros((6, 6))
        J[0:3, 0:3] = self.I
        J[3:6, 3:6] = self.m * np.eye(3)
        invJ = np.linalg.inv(J)

        self.dt = 0.02

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
        self.R = np.eye(6) * 1



    def controller_init(self):
        self.K0 = -ct.dlqr(self.A, self.B, self.Q, self.R)[0]
        self.K_ini = self.K0
        self.B_ini = self.B

    def set_vel(self, v, w):
        self.v = v
        self.w = w

    def reset(self):
        self.system_init()
        self.controller_init()
def evaluation(isPlot=False):
    linear_vel = np.array([2, 1, 0.4])
    angular_vel = np.array([1, 0, 1])
    config = {'type': TrajType.CONSTANT,
              'param': {'start_state': np.array([0, 0, 0, 0, 0, 0, 1]),
                        'linear_vel': linear_vel,
                        'angular_vel': angular_vel,
                        'dt': 0.02,
                        'nTraj': 300}}
    planner = SE3Planner(config)
    ref_SE3, ref_twist, dt = planner.get_traj()
    simulator = SingleRigidBodySimulator(dt)
    init_pos = np.array([0.4, 0, 0])
    init_quat = np.array([0, 0, 0, 1])
    init_state = np.hstack([init_pos, init_quat, np.zeros(6)])

    simulator.set_init_state(init_state)

    # container
    state_container = np.zeros((13, np.size(ref_SE3, 1)))
    state_container[:, 0] = init_state
    ctrl_container = np.zeros((6, np.size(ref_SE3, 1)))
    error_container = np.zeros((12, np.size(ref_SE3, 1)))

    # ctrl
    lti = LinearSE3ErrorDynamics()
    print(lti.K0)
    lti.set_vel(linear_vel, angular_vel)
    lti.reset()
    K = lti.K0
    print("-------------------")
    print(K)
    for i in range(np.size(ref_SE3, 1)):
        pos = simulator.curr_state[0:3]
        quat = simulator.curr_state[3:3 + 4]
        curr_SE3 = SE3(pos, quat)
        curr_omega_vel = simulator.curr_state[7:7 + 3 + 3]
        state_container[:, i] = simulator.curr_state

        curr_SE3_ref = SE3(ref_SE3[0:3, i], ref_SE3[3:3 + 4, i])
        ref_omega_vel = ref_twist[:, i]

        x = np.zeros(12)
        x_log = curr_SE3_ref.between(curr_SE3).log().coeffs()
        x[0:3] = x_log[3: 3 + 3]
        x[3:6] = x_log[0: 0 + 3]
        x[6:12] = curr_omega_vel - ref_omega_vel
        u = K.dot(x)
        ctrl_container[:, i] = u
        simulator.step(u)

    if isPlot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(ref_SE3[0, :], ref_SE3[1, :], ref_SE3[2, :], 'b')
        ax.plot(state_container[0, :], state_container[1, :], state_container[2, :], 'r')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return state_container




if __name__ == '__main__':
    evaluation(True)

