"""Turtlebot environment using PyBullet physica.
wrapper for SE3 and SO3 from liecasadi

"""
import os

import time
from liecasadi import SO3, SO3Tangent
import casadi as ca
import numpy as np
import pybullet as p
import pybullet_data
from environments.wheeled_mobile_robot.wheeled_mobile_robot_base import WheeledMobileRobot
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from utils.symbolic_system import FirstOrderModel
from gymnasium import spaces
from functools import partial
from utils.enum_class import CostType, DynamicsType

class Turtlebot3D(Turtlebot):
    NAME = 'turtlebot3d'
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'turtlebot.urdf')

    def __init__(self, init_state: np.ndarray = None, gui: bool = False, debug: bool = False, **kwargs):
        super().__init__(init_state=init_state, gui=gui, debug=debug, **kwargs)
        self.nState = 7 # pos and quat
        self.nControl = 6 # linear and angular velocity

    def get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.PYB_CLIENT)
        vel, ang_vel = p.getBaseVelocity(self.robot, physicsClientId=self.PYB_CLIENT)
        p.getLin
        return np.array(pos + quat + vel + ang_vel)


if __name__ == '__main__':
    env = Turtlebot3D(init_state=np.array([0, 0, 0]), gui=True)
    env.get_state()
    env.reset()
    time.sleep(5)
