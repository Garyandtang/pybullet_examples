import numpy as np
from manifpy import SE2, SO2, SE2Tangent, SO2Tangent
from manifpy import SE3, SO3, SE3Tangent, SO3Tangent
from utilsStuff.utils import skew

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
        omega = self.curr_state[7:7+3]
        vel = self.curr_state[10:10+3]
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
        next_state = np.hstack([next_pos, next_quat, next_omega, next_vel])
        self.curr_state = next_state
        return next_state

    # def step_2(self, u):
    #     pos = self.curr_state[0:3]
    #     quat = self.curr_state[3:3+4]
    #     omega = self.curr_state[7:7+3]
    #     vel = self.curr_state[10:10+3]
    #
    #     tau = u[0:3]
    #     f = u[3:3+3]
    #
    #     R = SO3(quat).rotation()
    #     pos_dot = R @ vel
    #     Omega = np.array([0, -omega[0], -omega[1], -omega[2],
    #                       omega[0], 0, omega[2], -omega[1],
    #                       omega[1], -omega[2], 0, omega[0],
    #                       omega[2], omega[1], -omega[0], 0]).reshape(4, 4)
    #     quat_dot = 0.5 * Omega @ quat
    #     omega_dot = np.linalg.inv(self.I).dot(tau - skew(omega).dot(self.I).dot(omega))
    #     vel_dot = - skew(omega).dot(vel) + f / self.m
    #
    #     next_pos = pos + pos_dot * self.dt
    #     next_quat = quat + quat_dot * self.dt
    #     next_quat /= np.linalg.norm(next_quat)
    #     next_omega = omega + omega_dot * self.dt
    #     next_vel = vel + vel_dot * self.dt
    #     next_state = np.hstack([next_pos, next_quat, next_omega, next_vel])
    #     self.curr_state = next_state
    #     return next_state


if __name__ == '__main__':
    w = np.array([1, 0, 0])
    v = np.array([1, 2, 3])
    w_cross_v  = np.cross(w, v)
    w_skew = skew(w)
    print(w_skew @ v)
    print(w_cross_v)
    # dt = 0.02
    # simulator = SE3Simulator(dt)
    # init_state = np.array([0, 0, 0,
    #                         0, 0, 0, 1,
    #                         0, 0, 3,
    #                         1, 0, 0])
    # simulator.set_init_state(init_state)
    # control = np.array([0, 0, 0, 0, 0, 0])
    # state_container = init_state
    # for i in range(200):
    #     simulator.step(control)
    #     state_container = np.vstack((state_container, simulator.curr_state))
    #
    # import matplotlib.pyplot as plt
    # # 3d plot x y z
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot(state_container[:, 0], state_container[:, 1], state_container[:, 2])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()
    #
    # # plot x y
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.plot(state_container[:, 0], state_container[:, 1])
    # plt.show()
