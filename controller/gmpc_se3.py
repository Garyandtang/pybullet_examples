import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from manifpy import SO3, SO3Tangent, SE3Tangent, SE3
from utils.enum_class import TrajType, ControllerType
from planner.ref_traj_generator import TrajGenerator
from planner.SE3_planner import SE3Planner
import manifpy as manif
import time
import pybullet as p
from utils.utils import skew

def adjoint(twist):
    omega = twist[0:3]
    v = twist[3:6]
    omega_hat = skew(omega)
    v_hat = skew(v)
    adj = np.zeros((6, 6))
    adj[0:3, 0:3] = omega_hat
    adj[3:6, 3:6] = omega_hat
    adj[3:6, 0:3] = v_hat
    return adj

def coadjoint(twist):
    omega = twist[0:3]
    v = twist[3:6]
    omega_hat = skew(omega)
    v_hat = skew(v)
    coadj = np.zeros((6, 6))
    coadj[0:3, 0:3] = -omega_hat
    coadj[3:6, 3:6] = -omega_hat
    coadj[0:3, 3:6] = -v_hat
    return coadj

def gamma_right(I, twist):
    omega = twist[0:3]
    v = twist[3:6]
    I_omega_hat = skew(I @ omega)
    v_hat = skew(v)
    res = np.zeros((6, 6))
    res[0:3, 0:3] = I_omega_hat
    res[3:6, 0:3] = v_hat
    res[0:3, 3:6] = v_hat
    return res

class GeometricMPC:
    def __init__(self, ref_traj_config):
        self.controllerType = ControllerType.SE3MPC
        self.nPos = 3  # [x, y, z]
        self.nQuat = 4  # [qx, qy, qz, qw]
        self.nTwist = 6  # [wx, wy, wz, vx, vy, vz]
        self.nControl = 6  #
        self.nTraj = None
        self.dt = None
        self.solve_time = 0.0
        self.setup_solver()
        self.set_ref_traj(ref_traj_config)
        self.I = np.eye(3)
        self.m = 1



    def set_ref_traj(self, traj_config):
        traj_generator = TrajGenerator(traj_config)
        self.ref_state, self.ref_twist, self.dt = traj_generator.get_traj()
        self.nTraj = self.ref_state.shape[1]

    def setup_solver(self, Q=[10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1], R=0.1, N=10):
        self.Q = np.diag(Q)
        self.R = R * np.diag(np.ones(self.nControl))
        self.N = N



    def solve(self, current_state, t):
        """
        current_state: [p, q, w, v] -> [x, y, z, qx, qy, qz, qw, wx, wy, wz, vx, vy, vz]
        """

        start_time = time.time()
        if self.ref_state is None:
            raise ValueError('Reference trajectory is not set up yet!')

        # get reference state and twist
        k = round(t / self.dt)
        curr_ref = self.ref_state[:, k]
        curr_ref_twist = self.ref_twist[:, k]

        curr_pos = current_state[0:3]
        curr_quat = current_state[3:7]
        curr_omega = current_state[7:10]
        curr_vel = current_state[10:13]
        curr_twist = np.hstack((curr_omega, curr_vel))
        curr_SE3 = SE3(curr_pos, curr_quat)
        ref_SE3 = SE3(curr_ref[0:3], curr_ref[3:7])
        psi_init = ref_SE3.between(curr_SE3).log().coeffs()  # position error, orientation error
        twist = curr_twist
        x_init = np.hstack((psi_init, twist))



        Q = self.Q
        R = self.R
        N = self.N
        dt = self.dt
        I_inv = np.linalg.inv(self.I)
        J = np.zeros((6, 6))
        J[0:3, 0:3] = self.I
        J[3:6, 3:6] = self.m * np.eye(3)
        J_inv = np.zeros((6, 6))
        J_inv[0:3, 0:3] = I_inv
        J_inv[3:6, 3:6] = np.eye(3) / self.m

        # setup casadi solver
        opti = ca.Opti()
        x_var = opti.variable(12, N+1) # [psi, twist] not delta twist
        u_var = opti.variable(6, N)  # [tau, f]

        # initial condition
        opti.subject_to(x_var[:, 0] == x_init)

        # dynamics constraints
        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            ref_twsit = self.ref_twist[:, index]
            adj = adjoint(ref_twsit)
            coadj = coadjoint(ref_twsit)
            temp = gamma_right(self.I, ref_twsit)
            H =  J_inv @ coadj @ J + J_inv @ temp
            A = np.zeros((12, 12))
            A[0:6, 0:6] = -adj # -adjoint
            A[0:6, 6:12] = np.eye(6)
            A[6:12, 6:12] = H
            B = np.zeros((12, 6))
            B[6:12, :] = J_inv
            x_next = x_var[:, i] + dt * (A @ x_var[:, i] + B @ u_var[:, i])
            opti.subject_to(x_var[:, i+1] == x_next)

        # cost function
        cost = 0
        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            cost += ca.mtimes([x_var[:, i].T, Q, x_var[:, i]]) + ca.mtimes(
                [u_var[:, i].T, R, u_var[:, i]])
        cost += ca.mtimes([x_var[:, N].T, 100*Q, x_var[:, N]])

        # control bound
        # todo

        opts_setting = { 'printLevel': 'none'}
        opti.minimize(cost)
        opti.solver('qpoases',opts_setting)
        sol = opti.solve()
        psi_sol = sol.value(x_var)
        u_sol = sol.value(u_var)
        end_time = time.time()
        self.solve_time = end_time - start_time
        return u_sol[:, 0]




    def get_solve_time(self):
        return self.solve_time

    def vel_cmd_to_local_twist(self, vel_cmd):
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def local_twist_to_vel_cmd(self, local_vel):
        return ca.vertcat(local_vel[0], local_vel[2])

    @property
    def get_controller_type(self):
        return self.controllerType



if __name__ == '__main__':
    a = np.array([1, 2, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    b = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    SE3_a = SE3(a[0:3], a[3:7])
    SE3_b = SE3(b[0:3], b[3:7])
    twist = np.array([1,2,3,4,5,6])
    print("self x")
    print(adjoint(twist))

    print("coadjoint")
    print(coadjoint(twist))

    print("-----------")
    temp = SE3Tangent(twist)
    adj = temp.smallAdj()
    print("adj: \n", adj)

    adj_trans = temp.smallAdj().T
    print("adj trans: \n", adj_trans)





