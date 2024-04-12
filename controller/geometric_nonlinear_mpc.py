import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from liecasadi import SE3, SE3Tangent, SO3, SO3Tangent
import math
from planner.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType, CostType, DynamicsType
from controllers.lqr.lqr_utils import *
import casadi as ca
from utils.enum_class import CostType, DynamicsType, ControllerType
from utils.symbolic_system import FirstOrderModel
from liecasadi import SO3
import time

"""
Geometric nonlinear MPC for unicycle model
"""

class GeoUnicycleModel:
    def __init__(self, config: dict = {}, **kwargs):
        self.nPos = 3
        self.nQuat = 4
        self.nTwist = 6


        # setup configuration
        self.config = config

    def set_dt_model(self, dt):
        print("Setting up geometric nonlinear unicycle dynamics")
