"""Turtlebot environment using PyBullet physica.


"""
import os

import time
from liecasadi import SO3, SO3Tangent
import casadi as ca
import numpy as np
import pybullet as p
import pybullet_data

from utils.symbolic_system import FirstOrderModel

from envs.base_env import BaseEnv
from functools import partial
from utils.enum_class import CostType, DynamicsType

class Turtlebot():
    NAME = 'turtlebot'
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'turtlebot.urdf')

    def __init__(self,
                 init_state: np.ndarray = None,
                 gui: bool = False,
                 **kwargs):
        # super().__init__(gui=gui, **kwargs)
        # configuration setup
        self.GUI = gui
        self.CTRL_FREQ = 50  # control frequency
        self.PYB_FREQ = 50  # simulator fr
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ

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
        # turtlebot model parameters
        self.length = 0.23  # length of the turtlebot
        self.width = 0.036  # width of the wheel of the turtlebot
        if init_state is None:
            self.init_state = np.array([0, 0, 0])
        elif isinstance(init_state, np.ndarray):
            self.init_state = init_state
        else:
            raise ValueError('[ERROR] in turtlebot.__init__(), init_state, type: {}, size: {}'.format(type(init_state),
                                                                                                         len(init_state)))
        self.reset()

    def reset(self, seed=None):
        # reset the simulation
        self._set_action_space()
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

        return self.get_state()

    def get_state(self):
        # [x, y, theta]
        pos, quat = p.getBasePositionAndOrientation(self.turtlebot, physicsClientId=self.PYB_CLIENT)
        euler = p.getEulerFromQuaternion(quat) # [-pi, pi]
        self.state = np.array([pos[0], pos[1], euler[2]])
        return self.state

    def get_twist(self):
        # [v, w]
        vel = p.getBaseVelocity(self.turtlebot, physicsClientId=self.PYB_CLIENT)
        self.twist = np.array([vel[0][0], vel[1][2]])
        return self.twist

    def calc_twist(self, action):
        # action: [v_l, v_r] in m/s left and right wheel velocity
        # twist: [v, w] in m/s linear and angular velocity
        v_l = action[0]
        v_r = action[1]
        v = (v_l + v_r) / 2
        w = (v_r - v_l) / self.length
        twist = np.array([v, w])
        return twist

    def vel_cmd_to_action(self, vel_cmd):
        # vel_cmd: [v, w] in m/s linear and angular velocity
        # action: [v_l, v_r] in m/s left and right wheel velocity
        vel_cmd = self.saturate_vel_cmd(vel_cmd)
        v = vel_cmd[0]
        w = vel_cmd[1]
        v_l = v - self.length * w / 2
        v_r = v + self.length * w / 2
        v_l = v_l / self.width
        v_r = v_r / self.width
        action = np.array([v_l, v_r])
        return action


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

        self.draw_point(self.state)
        return self.state, None, None, None

    def _denormalize_action(self, action):
        """ converts a normalized action into a physical action, only need in RL-based action
        :param action (ndarray):
        :return: action (ndarray):
        """
        return action

    def draw_point(self, point):
        p.addUserDebugPoints(
            [[point[0], point[1], 0.12]], [[0.1, 0, 0]], pointSize=3, lifeTime=0.5)

    def _preprocess_control(self, action):
        raise NotImplementedError

    def _set_action_space(self):
        self.v_min = -0.22
        self.v_max = 0.22
        self.w_min = -2.84
        self.w_max = 2.84

    def get_vel_cmd_limit(self):
        return self.v_min, self.v_max, self.w_min, self.w_max

    def saturate_vel_cmd(self, vel_cmd):
        v, w = vel_cmd
        if v < self.v_min:
            v = self.v_min
        elif v > self.v_max:
            v = self.v_max
        if w < self.w_min:
            w = self.w_min
        elif w > self.w_max:
            w = self.w_max
        return np.array([v, w])

    def draw_ref_traj(self, ref_SE2):
        # ref_se2: [x, y, cos(theta), sin(theta)]
        ref_traj = np.zeros((3, ref_SE2.shape[1]))
        ref_traj[0:2, :] = ref_SE2[0:2, :]
        ref_traj[2, :] = 0.1
        for i in range(ref_SE2.shape[1] -1):
            p1 = ref_traj[:, i]
            p2 = ref_traj[:, i+1]
            p.addUserDebugLine(p1, p2, [1, 0, 0], 2, physicsClientId=self.PYB_CLIENT)
        return ref_traj

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
