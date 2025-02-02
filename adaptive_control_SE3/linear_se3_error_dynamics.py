import control as ct
from utilsStuff.lie_utils import *
from planner.SE3_planner import SE3Planner
from utilsStuff.enum_class import TrajType
from utilsStuff.utils import generate_random_positive_definite_matrix, is_positive_definite
from environments.numerical_simulator.single_rigid_body_simulator import SingleRigidBodySimulator
from manifpy import SE3
import matplotlib.pyplot as plt
class LinearSE3ErrorDynamics:
    def __init__(self, config=None):
        if config is None:
            config = {'fixed_param': False,
                      'I': np.array([[1, 0.2, 0.1],
                                         [0.2, 1, 0.2],
                                         [0.1, 0.2, 1]]),
                      'm': 2,
                      'v': np.array([2, 0, 0.2]),
                      'w': np.array([0, 0, 1])}
        self.fixed_param = config['fixed_param']
        I = config['I']
        m = config['m']
        self.v = config['v']
        self.w = config['w']

        self.system_init(I=I, m=m)
        self.controller_init()


    def system_init(self, I = np.array([[1, 0.2, 0.1],
                               [0.2, 1, 0.2],
                               [0.1, 0.2, 1]]), m = 2):
        if self.fixed_param:
            self.I = I
            self.m = m
        else:
            self.I = I + 0.1 * generate_random_positive_definite_matrix(3)
            self.m = m + 1 * np.random.rand()

        assert is_positive_definite(self.I)
        assert self.m > 0
        # reference control to counteract Coriolis force
        self.ud = np.zeros(6)
        self.ud[0:3] = np.array([0, 0, 0])
        self.ud[3:6] = self.m * skew(self.w).dot(self.v)
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

    def reset(self, config):
        self.fixed_param = config['fixed_param']
        I = config['I']
        m = config['m']
        self.v = config['v']
        self.w = config['w']

        self.system_init(I=I, m=m)
        self.controller_init()

    def update_controller(self, K):
        self.K0 = K
def evaluation(isPlot=False, lti=LinearSE3ErrorDynamics(), config=None, robot_init_state=None):
    linear_vel = lti.v
    angular_vel = lti.w
    config = {'type': TrajType.CONSTANT,
              'param': {'start_state': np.array([0, 0, 0, 0, 0, 0, 1]),
                        'linear_vel': linear_vel,
                        'angular_vel': angular_vel,
                        'dt': 0.02,
                        'nTraj': 300}}
    planner = SE3Planner(config)
    ref_SE3, ref_twist, dt = planner.get_traj()
    simulator = SingleRigidBodySimulator(dt)
    init_pos = np.array([0, 0, 0])
    init_quat = np.array([0, 0, 0, 1])
    if robot_init_state is None:
        init_state = np.hstack([init_pos, init_quat, angular_vel, linear_vel])
    else:
        init_state = robot_init_state

    simulator.set_init_state(init_state)

    # container
    state_container = np.zeros((13, np.size(ref_SE3, 1)))
    state_container[:, 0] = init_state
    ctrl_container = np.zeros((6, np.size(ref_SE3, 1)))

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
        x[0:3] = x_log[3: 3 + 3]  # log(R)
        x[3:6] = x_log[0: 0 + 3]  # log(p)
        x[6:12] = curr_omega_vel - ref_omega_vel
        u = lti.K0.dot(x) + lti.ud
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

    return state_container, ctrl_container




if __name__ == '__main__':
    lti_config = {'fixed_param': True,
                  'I': 2 * np.array([[1, 0.2, 0.1],
                                 [0.2, 1, 0.2],
                                 [0.1, 0.2, 1]]),
                  'm': 2 * 2,
                  'v': np.array([2, 0, 0.2]),
                  'w': np.array([0, 0, 1])}
    lti = LinearSE3ErrorDynamics(lti_config)
    evaluation(True, lti)


