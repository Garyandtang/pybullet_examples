"""Turtlebot environment using PyBullet physics.


"""
import os
import copy

import time
import xml.etree.ElementTree as etxml
from liecasadi import SO3, SO3Tangent


import casadi as cs
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from gym.utils import seeding
from utils.symbolic_system import FirstOrderModel
from utils import utils

from controllers.lqr.lqr import LQR
from utils.enum_class import Task
from envs.base_env import BaseEnv
from functools import partial

class Turtlebot(BaseEnv):
    NAME = 'turtlebot'
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'turtlebot.urdf')

    def __init__(self,
                 init_state: np.ndarray = None,
                 gui: bool = False,
                 **kwargs):
        super().__init__(gui=gui, **kwargs)

        # create a PyBullet physics simulation
        self.PYB_CLIENT = -1
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)

        # disable urdf auto-loading
        p.setPhysicsEngineParameter(enableFileCaching=1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # set gui and rendering size
        self.RENDER_HEIGHT = int(200)
        self.RENDER_WIDTH = int(320)

        # set the init state
        self.nState = 3
        self.nControl = 2
        if init_state is None:
            self.init_state = np.array([0, 0, 0])
        elif isinstance(init_state, np.ndarray):
            self.init_state = init_state
        else:
            raise ValueError('[ERROR] in turtlebot.__init__(), init_state, type: {}, size: {}'.format(type(init_state),
                                                                                                         len(init_state)))
        self.U_GOAL = np.zeros(self.nControl)
        self.X_GOAL = np.zeros(self.nState)

        self.reset()

    def reset(self, seed=None):
        super().before_reset(seed=seed)
        # reset the simulation
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.PYB_CLIENT)

        # turtlebot setting
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.PYB_CLIENT)
        self.turtlebot = p.loadURDF(self.URDF_PATH, self.init_state, physicsClientId=self.PYB_CLIENT)
        p.resetJointState(self.turtlebot, 0, 0, 0, physicsClientId=self.PYB_CLIENT)
        p.resetJointState(self.turtlebot, 1, 0, 0, physicsClientId=self.PYB_CLIENT)

        # turtlebot model parameters
        self.length = 0.23 # length of the turtlebot
        self.width = 0.025 # width of the wheel of the turtlebot

        self._setup_symbolic()

        return self.get_state()

    def get_state(self):
        # [x, y, theta]
        pos, quat = p.getBasePositionAndOrientation(self.turtlebot, physicsClientId=self.PYB_CLIENT)
        euler = p.getEulerFromQuaternion(quat) # [-pi, pi]
        self.state = np.array([pos[0], pos[1], euler[2]])
        return self.state

    def step(self, action):
        # action: [v_l, v_r] in m/s left and right wheel velocity
        v_l = action[0]
        v_r = action[1]
        # TODO: understand how to step per control
        for _ in range(self.PYB_STEPS_PER_CTRL):
            p.setJointMotorControl2(self.turtlebot, 0, p.VELOCITY_CONTROL, targetVelocity=v_l, force=1000,
                                    physicsClientId=self.PYB_CLIENT)
            p.setJointMotorControl2(self.turtlebot, 1, p.VELOCITY_CONTROL, targetVelocity=v_r, force=1000,
                                    physicsClientId=self.PYB_CLIENT)
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)

        self.state = self.get_state()

        return self.state, None, None, None

    def _denormalize_action(self, action):
        """ converts a normalized action into a physical action, only need in RL-based action
        :param action (ndarray):
        :return: action (ndarray):
        """
        return action

    def _preprocess_control(self, action):
        raise NotImplementedError

    def _set_action_space(self, state):
        raise NotImplementedError

    def _setup_symbolic(self):
        l = self.length
        w = self.width
        x, y, theta = cs.MX.sym('x'), cs.MX.sym('y'), cs.MX.sym('theta')
        v_l, v_r = cs.MX.sym('v_l'), cs.MX.sym('v_r')
        v = (v_l + v_r) / 2
        w = (v_r - v_l) / l
        x_dot = cs.cos(theta) * v
        y_dot = cs.sin(theta) * v
        theta_dot = w
        X = cs.vertcat(x, y, theta)
        X_dot = cs.vertcat(x_dot, y_dot, theta_dot)



        U = cs.vertcat(v_l, v_r)
        Y = X
        nx = 3
        nu = 2
        # define cost (quadratic form)
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        if 0:
            cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
            cost = {'cost_func': cost_func, 'vars': {'X': X, 'Xr': Xr, 'U': U, 'Ur': Ur, 'Q': Q, 'R': R}}
        else:
            pos_cost = 0.5 * (X[:2] - Xr[:2]).T @ Q[:2, :2] @ (X[:2] - Xr[:2])
            theta = X[2]
            theta_target = Xr[2]
            so3 = SO3.from_euler(cs.vertcat(0, 0, theta))
            so3_target = SO3.from_euler(cs.vertcat(0, 0, theta_target))
            quat_diff = 1 - cs.power(cs.mtimes(cs.transpose(so3.as_quat().coeffs()), so3_target.as_quat().coeffs()), 2)
            ori_cost = 0.5 * quat_diff.T @ Q[2:, 2:] @ quat_diff
            cost_func = pos_cost + ori_cost + 0.5 * (U - Ur).T @ R @ (U - Ur)
            cost = {'cost_func': cost_func, 'vars': {'X': X, 'Xr': Xr, 'U': U, 'Ur': Ur, 'Q': Q, 'R': R}}
        # define dyn and cost dictionaries
        first_dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}

        params = {
            'X_EQ': np.array([0, 0, 0, 0]),
            'U_EQ': np.zeros(self.nControl)  # np.atleast_2d(self.U_GOAL)[0, :],
        }
        self.symbolic = FirstOrderModel(first_dynamics, cost, self.CTRL_TIMESTEP, params)



if __name__ == '__main__':
    key_word = {'gui': False}
    env_func = partial(Turtlebot, **key_word)
    turtle_env = Turtlebot(gui=True)
    turtle_env.reset()
    while 1:
        state = turtle_env.step([1, 1])
        print(p.getJointState(1, 1)[1])
        print(state[0])
        print()
        time.sleep(0.02)
