import scipy.linalg
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import math
from ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType, CostType, DynamicsType
from controllers.lqr.lqr_utils import *
import casadi as ca
from utils.enum_class import CostType, DynamicsType
from utils.symbolic_system import FirstOrderModel
from liecasadi import SO3

"""
naive MPC for unicycle model
"""


class UnicycleModel:
    def __init__(self, config: dict = {}, pyb_freq: int = 50, **kwargs):
        self.nState = 3
        self.nControl = 2

        self.control_freq = pyb_freq
        self.dt = 1. / self.control_freq

        # setup configuration
        self.config = config
        if not config:
            self.config = {'cost_type': CostType.POSITION_EULER, 'dynamics_type': DynamicsType.EULER_FIRST_ORDER}
        if self.config['dynamics_type'] == DynamicsType.EULER_FIRST_ORDER:
            self.set_up_euler_first_order_dynamics()
        elif self.config['dynamics_type'] == DynamicsType.EULER_SECOND_ORDER:
            pass
        elif self.config['dynamics_type'] == DynamicsType.DIFF_FLAT:
            pass
        print(config.get("dynamics_type"))

    def set_up_euler_first_order_dynamics(self):
        print("Setting up Euler first order dynamics")
        nx = self.nState
        nu = self.nControl
        # state
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        theta = ca.MX.sym('theta')
        X = ca.vertcat(x, y, theta)
        # control

        v = ca.MX.sym('v')  # linear velocity
        w = ca.MX.sym('w')  # angular velocity
        U = ca.vertcat(v, w)

        # state derivative
        x_dot = ca.cos(theta) * v
        y_dot = ca.sin(theta) * v
        theta_dot = w
        X_dot = ca.vertcat(x_dot, y_dot, theta_dot)

        # cost function
        self.costType = self.config['cost_type']
        print("Cost type: {}".format(self.costType))
        Q = ca.MX.sym('Q', nx, nx)
        R = ca.MX.sym('R', nu, nu)
        Xr = ca.MX.sym('Xr', nx, 1)
        Ur = ca.MX.sym('Ur', nu, 1)
        if self.costType == CostType.POSITION:
            cost_func = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2]) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        elif self.costType == CostType.POSITION_QUATERNION:
            pos_cost = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2])
            theta = X[2]
            theta_target = Xr[2]
            so3 = SO3.from_euler(ca.vertcat(0, 0, theta))
            so3_target = SO3.from_euler(ca.vertcat(0, 0, theta_target))
            quat_diff = 1 - ca.power(ca.dot(so3.quat, so3_target.quat), 2)
            quat_cost = 0.5 * quat_diff.T @ Q[2:, 2:] @ quat_diff
            cost_func = pos_cost + quat_cost + 0.5 * (U - Ur).T @ R @ (U - Ur)
        elif self.costType == CostType.POSITION_EULER:
            pos_cost = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2])
            theta = X[2]
            theta_target = Xr[2]
            euler_diff = 1 - ca.cos(theta - theta_target)
            euler_cost = 0.5 * euler_diff.T @ Q[2:, 2:] @ euler_diff
            cost_func = pos_cost + euler_cost + 0.5 * (U - Ur).T @ R @ (U - Ur)
        elif self.costType == CostType.NAIVE:
            cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        else:
            raise ValueError('[ERROR] in turtlebot._setup_symbolic(), cost_type: {}'.format(self.costType))
        cost = {'cost_func': cost_func, 'vars': {'X': X, 'Xr': Xr, 'U': U, 'Ur': Ur, 'Q': Q, 'R': R}}

        # define dynamics and cost dict
        dynamics = {'dyn_eqn': X_dot, 'vars': {'X': X, 'U': U}}
        params = {
            'X_EQ': np.zeros(self.nState),  # np.atleast_2d(self.X_GOAL)[0, :],
            'U_EQ': np.zeros(self.nControl)  # np.atleast_2d(self.U_GOAL)[0, :],
        }
        self.symbolic = FirstOrderModel(dynamics, cost, self.dt, params)


class NaiveMPC:
    def __init__(self, ref_traj_config, model_config={}):
        config = model_config
        # dynamics
        self.model = UnicycleModel(config).symbolic
        self.nState = self.model.nx  # 3 (x, y, theta)
        self.nControl = self.model.nu  # 2 (v, w)
        self.set_ref_traj(ref_traj_config)
        self.set_solver()
        self.cost_func = self.model.cost_func

    def set_solver(self, q=[15, 15, 6], R=0.5, N=10):
        self.Q = 5*np.diag(q)
        self.R = R * np.eye(self.model.nu)
        self.N = N

    def set_ref_traj(self, traj_config):
        traj_generator = TrajGenerator(traj_config)
        ref_SE2, ref_twist, self.dt = traj_generator.get_traj()
        self.ref_state = np.zeros((self.nState, ref_SE2.shape[1]))
        # convert SE2 to x, y, theta
        for i in range(ref_SE2.shape[1]):
            self.ref_state[:2, i] = ref_SE2[:2, i]
            self.ref_state[2, i] = SE2(ref_SE2[:, i]).angle()

        # convert twist to v, w
        self.ref_control = np.zeros((self.nControl, ref_twist.shape[1]))
        for i in range(ref_twist.shape[1]):
            self.ref_control[0, i] = ref_twist[0, i]
            self.ref_control[1, i] = ref_twist[2, i]


        self.nTraj = self.ref_state.shape[1]

    def solve(self, state, t):
        """
        state: [x, y, theta]
        t: time -> index of reference trajectory (t = k * dt)
        """
        if self.ref_state is None:
            raise ValueError('Reference trajectory is not set up yet!')

        nu = self.nControl
        nx = self.nState
        k = math.ceil(t / self.dt)
        N = self.N
        index_end = min(k + N, self.nTraj - 1)
        X = self.ref_state[:, index_end]
        # x_goal = np.array([X.x(), X.y(), X.angle()])
        opti = ca.Opti()
        x_var = opti.variable(nx, N + 1)
        u_var = opti.variable(nu, N)

        # initial state constraint
        opti.subject_to(x_var[:, 0] == state)

        # dynamics constraint
        for i in range(N):
            # Euler first order
            x_next = x_var[:, i] + self.dt * self.model.fc_func(x_var[:, i], u_var[:, i])
            opti.subject_to(x_var[:, i + 1] == x_next)

        # cost function
        cost = 0

        for i in range(N):
            index = min(k + i, self.nTraj - 1)
            x_target = self.ref_state[:, index]
            u_target = self.ref_control[:, index]
            cost += self.cost_func(x_var[:, i], x_target, u_var[:, i], u_target, self.Q, self.R)

        # cost
        opti.minimize(cost)
        opti.solver('ipopt')
        sol = opti.solve()
        u = sol.value(u_var[:, 0])
        return u

    def _to_local_vel(self, vel_cmd):
        return ca.vertcat(vel_cmd[0], 0, vel_cmd[1])

    def _to_vel_cmd(self, local_vel):
        return ca.vertcat(local_vel[0], local_vel[2])


def test_mpc():
    init_state = np.array([-0.2, -0.2, 0])
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.array([0, 0, 0]),
                             'linear_vel': 0.5,
                             'angular_vel': 0.5,
                             'nTraj': 170,
                             'dt': 0.05}}
    ref_traj_generator = TrajGenerator(traj_config)
    ref_SE2, ref_twist, dt = ref_traj_generator.get_traj()
    model_config = {'cost_type': CostType.POSITION,
              'dynamics_type': DynamicsType.EULER_FIRST_ORDER}
    mpc = NaiveMPC(traj_config, model_config=model_config)

    t = 0
    # contrainer to store state
    state_store = np.zeros((3, mpc.nTraj))
    state_store[:, 0] = init_state
    dyn = mpc.model.fc_func
    # start simulation
    for i in range(mpc.nTraj - 1):
        state = state_store[:, i]
        xi = mpc.solve(state, t)
        state = state + mpc.dt * dyn(state, xi)
        state = state.full().reshape((3,))
        state[2] = wrap_angle(state[2])
        state_store[:, i + 1] = state
        t += mpc.dt

    # plot
    plt.figure()
    plt.plot(ref_SE2[0, :], ref_SE2[1, :], 'r')
    plt.plot(state_store[0, :], state_store[1, :], 'b')
    plt.legend(['reference', 'trajectory'])

    plt.show()

    # plot distance difference
    distance_store = np.linalg.norm(state_store[0:2, :] - ref_SE2[0:2, :], axis=0)
    plt.figure()
    plt.plot(distance_store)
    plt.title('distance difference')
    plt.show()

    # plot orientation difference
    orientation_store = np.zeros(mpc.nTraj)
    for i in range(mpc.nTraj):
        X_d = SE2(ref_SE2[:, i])
        X = SE2(state_store[0, i], state_store[1, i], state_store[2, i])
        X_d_inv_X = SO2(X_d.angle()).between(SO2(X.angle()))
        orientation_store[i] = scipy.linalg.norm(X_d_inv_X.log().coeffs())

    plt.figure()
    plt.plot(orientation_store[0:])
    plt.title('orientation difference')

    plt.show()


def wrap_angle(angle):
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    return angle


if __name__ == '__main__':
    # test_generate_ref_traj()
    test_mpc()
