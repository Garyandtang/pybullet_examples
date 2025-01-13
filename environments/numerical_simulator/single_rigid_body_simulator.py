import numpy as np
from manifpy import SE2, SO2, SE2Tangent, SO2Tangent
from manifpy import SE3, SO3, SE3Tangent, SO3Tangent
from utilsStuff.utils import skew
from planner.SE3_planner import SE3Planner


class SingleRigidBodySimulator:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.nState = 13 # x,q,v,w
        self.nControl = 6 # [f,tau]
        self.curr_state = np.array([0, 0, 0,
                                    0, 0, 0, 1,
                                    0, 0, 0,
                                    0, 0, 0])

        self.m = 1.0
        self.I = np.eye(3)
        self.g = np.array([0, 0, 0])

    def set_init_state(self, init_state):
        quat = init_state[3:3+4]
        # check quat is normalized
        assert np.isclose(np.linalg.norm(quat), 1)
        self.curr_state = init_state

    def step(self, u):
        pos = self.curr_state[0:3]
        quat = self.curr_state[3:3+4]
        vel = self.curr_state[7:7+3]
        omega = self.curr_state[10:10+3]
        twist = np.hstack([vel, omega])

        tau = u[0:3]
        f = u[3:3+3]

        curr_SE3 = SE3(pos, quat)
        next_SE3 = curr_SE3 * SE3Tangent(twist * self.dt).exp()
        next_pos = next_SE3.translation()
        next_quat = next_SE3.quat()
        omega_dot = np.linalg.inv(self.I).dot(tau - skew(omega).dot(self.I).dot(omega))
        vel_dot = f / self.m - skew(omega).dot(vel)
        next_omega = omega + omega_dot * self.dt
        next_vel = vel + vel_dot * self.dt
        next_state = np.hstack([next_pos, next_quat, next_vel, next_omega])
        self.curr_state = next_state
        return next_state



def test_simulator():
    dt = 0.02
    simulator = SingleRigidBodySimulator(dt)
    init_state = np.array([0, 0, 0,
                            0, 0, 0, 1,
                            2, 0, 0.2,
                            0, 0, 1])
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

if __name__ == '__main__':
    test_simulator()