import numpy as np
from manifpy import SE2, SO2, SE2Tangent, SO2Tangent
from manifpy import SE3, SO3, SE3Tangent, SO3Tangent
from utilsStuff.utils import skew
from planner.SE3_planner import SE3Planner
from utilsStuff.enum_class import TrajType
import matplotlib.pyplot as plt
# from adaptive_control_SE3.linear_se3_error_dynamics import LinearSE3ErrorDynamics


class SingleRigidBodySimulator:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.nState = 13 # x,q, w, v
        self.nControl = 6 # [f,tau]
        self.curr_state = np.array([0, 0, 0,
                                    0, 0, 0, 1,
                                    0, 0, 0,
                                    0, 0, 0])

        self.m = 1.0 * 2
        self.I = np.eye(3) * 6
        self.g = np.array([0, 0, 0])

        self.I = np.array([[1, 0.2, 0.1],
                            [0.2, 1, 0.2],
                            [0.1, 0.2, 1]])

    def set_init_state(self, init_state):
        quat = init_state[3:3+4]
        # check quat is normalized
        assert np.isclose(np.linalg.norm(quat), 1)
        self.curr_state = init_state

    def step(self, u):
        pos = self.curr_state[0:3]
        quat = self.curr_state[3:3+4]
        omega = self.curr_state[7:7+3]
        vel = self.curr_state[10:10+3]
        twist = np.hstack([vel, omega])

        tau = u[0:3]
        f = u[3:3+3]


        curr_SE3 = SE3(pos, quat)
        next_SE3 = curr_SE3 + SE3Tangent(twist * self.dt)
        next_pos = next_SE3.translation()
        next_quat = next_SE3.quat()
        omega_dot = np.linalg.inv(self.I).dot(tau - skew(omega).dot(self.I).dot(omega))
        # todo check this
        vel_dot = f / self.m - skew(omega).dot(vel)
        next_omega = omega + omega_dot * self.dt
        next_vel = vel + vel_dot * self.dt
        next_state = np.hstack([next_pos, next_quat, next_omega, next_vel])
        self.curr_state = next_state
        return next_state

    def get_true_I_m(self):
        return self.I, self.m


def test_simulator():
    dt = 0.02
    simulator = SingleRigidBodySimulator(dt)
    init_state = np.array([0, 0, 0,
                            0, 0, 0, 1,
                           0, 0, 1,
                            2, 0, 0.2]
                            )
    simulator.set_init_state(init_state)
    control = np.array([0, 0, 0, 0, 0, 0])
    print(simulator.step(control))
    state_container = init_state
    for i in range(300):
        simulator.step(control)
        state_container = np.vstack((state_container, simulator.curr_state))

    import matplotlib.pyplot as plt
    # 3d plot x y z
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state_container[:, 0], state_container[:,1], state_container[:, 2], 'b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def test_tracking_ctrl():
    linear_vel = np.array([0, 0, 0.2])
    angular_vel = np.array([1, 1, 0])
    config = {'type': TrajType.CONSTANT,
              'param': {'start_state': np.array([0, 0, 0, 0, 0, 0, 1]),
                        'linear_vel': linear_vel,
                        'angular_vel': angular_vel,
                        'dt': 0.02,
                        'nTraj': 3000}}
    planner = SE3Planner(config)
    ref_SE3, ref_twist, dt = planner.generate_constant_traj(config['param'])
    print(ref_SE3.shape)
    simulator = SingleRigidBodySimulator(dt)
    init_pos = np.array([0.4, 0, 0])
    init_quat = np.array([0, 0, 0, 1])
    init_state = np.hstack([init_pos, init_quat, angular_vel, linear_vel])
    # init_state = np.array([0.4, 0, 0,
    #                         0, 0, 0, 1,
    #                         0, 0, 0,
    #                         0, 0, 0])
    simulator.set_init_state(init_state)

    # container
    state_container = np.zeros((13, np.size(ref_SE3, 1)))
    state_container[:, 0] = init_state
    ctrl_container = np.zeros((6, np.size(ref_SE3, 1)))
    error_container = np.zeros((12, np.size(ref_SE3, 1)))

    # ctrl
    from adaptive_control_SE3.linear_se3_error_dynamics import LinearSE3ErrorDynamics
    lti = LinearSE3ErrorDynamics()
    lti.set_vel(linear_vel, angular_vel)
    lti.reset()
    K = lti.K0
    print(K)
    for i in range(np.size(ref_SE3, 1)):
        pos = simulator.curr_state[0:3]
        quat = simulator.curr_state[3:3+4]
        curr_SE3 = SE3(pos, quat)
        curr_omega_vel = simulator.curr_state[7:7+3+3]
        state_container[:, i] = simulator.curr_state

        curr_SE3_ref = SE3(ref_SE3[0:3, i], ref_SE3[3:3+4, i])
        ref_omega_vel = ref_twist[:, i]

        x = np.zeros(12)
        x_log = curr_SE3_ref.between(curr_SE3).log().coeffs()
        x[0:3] = x_log[3: 3 + 3]
        x[3:6] = x_log[0: 0 + 3]
        x[6:12] = curr_omega_vel - ref_omega_vel
        print("x: ", x)
        u = K.dot(x)
        ctrl_container[:, i] = u
        simulator.step(u)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ref_SE3[0, :], ref_SE3[1, :], ref_SE3[2, :], 'b')
    ax.plot(state_container[0, :], state_container[1, :], state_container[2, :], 'r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == '__main__':
    test_tracking_ctrl()