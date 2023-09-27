import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from liecasadi import SO3, SO3Tangent, SE3Tangent, SE3
from utils.enum_class import TrajType, ControllerType
from controller.ref_traj_generator import TrajGenerator
import time

class SE3MPC:
    def __init__(self):
        self.controllerType = ControllerType.SE3MPC
        self.nPos = 3  # [x, y, z]
        self.nQuat = 4  # [qw, qx, qy, qz]
        self.nTwist = 6  # [vx, vy, vz, wx, wy, wz]
        self.nControl = None # todo: not implemented yet
        self.nTraj = None
        self.dt = None
        self.nPred = None
        self.solve_time = 0.0
        self.setup_solver()
        self.set_ref_traj()

    def set_ref_traj(self, traj_config):
        traj_generator = TrajGenerator(traj_config)
        self.ref_state, self.ref_control, self.dt = traj_generator.get_traj()
        self.nTraj = self.ref_state.shape[1]

    def setup_solver(self, Q, R, N):
        raise NotImplementedError

    def set_control_bound(self, v_min, v_max, w_min, w_max):
        raise NotImplementedError

    def solve(self, curr_state, t):
        start_time =time.time()
        if self.ref_state is None:
            raise ValueError('Reference trajectory is not set up yet!')
        dt = self.dt
        k = round(t / dt)
        R = self.R
        Q = self.Q


        # setup casadi solver
        opti = cs.Opti()
        pos = opti.variable(self.nPos, self.nPred)
        quat = opti.variable(self.nQuat, self.nPred)
        twist = opti.variable(self.nTwist, self.nPred - 1)

        # initial condition
        curr_pos = curr_state[:3]  # [x, y, z]
        curr_quat = curr_state[3:]  # [qw, qx, qy, qz]
        opti.subject_to(pos[:, 0] == curr_pos)
        opti.subject_to(quat[:, 0] == curr_quat)

        # dynamics constraints
        for i in range(self.nPred - 1):
            curr_SE3 = SE3(pos[:, i], quat[:, i])
            next_SE3 = SE3(pos[:, i + 1], quat[:, i + 1])
            curr_se3 = SE3Tangent(twist[:, i]*dt)
            forward_SE3 = curr_SE3 * curr_se3.exp()
            opti.subject_to(forward_SE3.pos() == next_SE3.pos())
            opti.subject_to(forward_SE3.quat() == next_SE3.quat())

        # cost function
        cost = 0
        for i in range(self.nPred - 1):
            index = min(k + i, self.nTraj - 1)
            curr_SE3 = SE3(pos[:, i], quat[:, i])
            ref_SE3 = SE3(self.ref_state[:3, index], self.ref_state[3:, index])
            SE3_diff = curr_SE3 - ref_SE3  # Log(SE3_ref^-1 * SE3), vector space
            cost += cs.mtimes([SE3_diff.vector().T, Q, SE3_diff.vector()])
            twist_d = np.zeros(6)
            cost += cs.mtimes([(twist[:, i] - twist_d).T, R, (twist[:, i] - twist_d)])

        last_SE3 = SE3(pos[:, -1], quat[:, -1])
        last_ref_SE3 = SE3(self.ref_state[:3, -1], self.ref_state[3:, -1])
        last_SE3_diff = last_SE3 - last_ref_SE3
        cost += cs.mtimes([last_SE3_diff.vector().T, Q, last_SE3_diff.vector()])

        opti.minimize(cost)

        # solve
        p_opts = {"expand": True}
        s_opts = {"max_iter": 100}
        opti.solver('ipopt', p_opts, s_opts)
        sol = opti.solve()
        self.solve_time = time.time() - start_time

        return sol.value(twist[:, 0])







