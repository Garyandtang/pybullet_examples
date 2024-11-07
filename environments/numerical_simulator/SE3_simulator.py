import numpy as np
from manifpy import SE2, SO2, SE2Tangent, SO2Tangent
from manifpy import SE3, SO3, SE3Tangent, SO3Tangent
from utils.utils import skew

# SE3 force control simulator
class SE3Simulator:
    def __init__(self, dt=0.02, I=np.eye(3), g=np.array([0, 0, 0]), m=1.0):
        self.dt = dt
        self.m = m
        self.I = I
        self.g = g
        self.nWrench = 6  # [F, tau]
        self.nState = 13  # [p, q, v, w]
        self.curr_state = np.array([0, 0, 0,
                                    0, 0, 0, 1,
                                    0, 0, 0,
                                    0, 0, 0])

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

        f = u[0:3]
        tau = u[3:3+3]

        curr_SE3 = SE3(pos, quat)
        next_SE3 = curr_SE3 * SE3Tangent(twist * self.dt).exp()
        next_pos = next_SE3.translation()
        next_quat = next_SE3.rotation().coeffs()
        next_vel = vel + (-skew(omega) @ vel + f / self.m + self.g) * self.dt

        curr_SE2 = SE2(self.curr_state[0], self.curr_state[1], self.curr_state[2])
        next_SE2 = curr_SE2 * SE2Tangent(twist * self.dt).exp()
        self.curr_state = np.array([next_SE2.x(), next_SE2.y(), next_SE2.angle()])
        return self.curr_state


if __name__ == '__main__':
    state = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    pos = state[0:3]
    quat = state[3:3+4]
    vel = state[7:7+3]
    omega = state[10:10+3]
    twsit = np.hstack([vel, omega])
    dt = 0.02
    SE3_X = SE3(pos, quat)
    Rot = SE3_X.rotation()

    # w_hat = ca.skew(omega)


    print(w_hat)