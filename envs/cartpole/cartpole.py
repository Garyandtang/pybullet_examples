'''Cartpole environment using PyBullet physics.

Classic cart-pole system implemented by Rich Sutton et al.
    * http://incompleteideas.net/sutton/book/code/pole.c

Also see:
    * github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    * github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/cartpole_bullet.py
    * https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/envs/gym_control/cartpole.py
'''
import abc
import os
import copy
import math
import time
import xml.etree.ElementTree as etxml
from copy import deepcopy
import gymnasium as gym
import casadi as cs
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from abc import ABC, abstractmethod
from gym.utils import seeding
from symbolic_system import FirstOrderModel
from utils import utils

from controllers.lqr.lqr import LQR
from utils.enum_class import Task

# from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
# from safe_control_gym.math_and_models.normalization import normalize_angle

def init_client():
    physics_client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(0)  # not real time
    p.setTimeStep(0.02)
    client_inited = True
    return client_inited



class BaseEnv(gym.Env, abc.ABC):
    _count = 0
    NAME = 'base'

    def __init__(self,
                 gui: bool = False,
                 seed=None,
                 task: Task = Task.STABILIZATION,
                 randomized_init: bool = True,
                 pyb_freq: int = 50,
                 ctrl_freq: int = 50,
                 output_dir: str = None,
                 **kwargs):
        self.idx = self.__class__._count
        self.__class__._count += 1
        self.GUI = gui
        self.Task = task
        self.CTRL_FREQ = ctrl_freq  # control frequency
        self.PYB_FREQ = pyb_freq  # simulator frequency
        # simulator frequency should larger than control frequency and should be divisible by ctrl one
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseEnv.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        self.RANDOMIZED_INIT = randomized_init
        if output_dir is None:
            output_dir = os.getcwd()
        self.output_dir = output_dir
        self.np_random, seed = seeding.np_random(seed)
        self.seed(seed)

        self.inited = False
        self.in_reset = False
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0
        self.current_raw_action = None
        self.current_physical_action = None  # current_raw_action unnormalized if it was normalized
        self.current_noisy_physical_action = None  # current_physical_action with noise added
        self.current_clipped_action = None  # current_noisy_physical_action clipped to physical action bounds

        self._set_action_space()

    def before_reset(self, seed=None):
        self.inited = True
        self.in_reset = True

    def before_step(self, action):
        """

        :param action (ndarray/scalar): The raw action returned by the controller.
        :return: action (ndarray): the processed action to be executed
        """
        if not self.inited:
            raise ValueError('[ERROR]: not init before call step().')

        action = np.atleast_1d(action)
        if action.ndim != 1 or action[0] is None:
            raise ValueError('[ERROR]: The action returned by the controller must be 1 dimensional.')

        self.current_raw_action = action

        processed_action = self._preprocess_control(action)

        return processed_action

    @abstractmethod
    def _preprocess_control(self, action):
        raise NotImplementedError

    @abstractmethod
    def _denormalize_action(self, action):
        """converts a normalized action into a physical action, only need in RL-based action
        :param action (ndarray):
        :return: action (ndarray):
        """
        raise NotImplementedError

    @abstractmethod
    def _set_action_space(self):
        """
        Defines the action space of the environment
        :return: None
        """
        raise NotImplementedError

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        # todo: seed for action space and disturbs
        # self.action_space.seed(seed)
        # for _, disturbs in self.disturbances.items():
        #     disturbs.seed(self)
        return [seed]


class CartPole(BaseEnv):
    NAME = 'cartpole'
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'cartpole_template.urdf')

    INIT_STATE_RAND_INFO = {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_theta_dot': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        }
    }

    def __init__(self,
                 init_state=None,
                 inertial_prop=None,
                 # custom args
                 obs_goal_horizon=0,
                 obs_wrap_angle=False,
                 rew_state_weight=1.0,
                 rew_act_weight=0.0001,
                 rew_exponential=True,
                 done_on_out_of_bound=True,
                 **kwargs):
        ''' Initialize a cartpole environment:
        :param init_state:
        :param inertial_prop:
        :param obs_goal_horizon:
        :param obs_warp_angle:
        :param rew_state_weight:
        :param rew_act_weight:
        :param rew_exponential:
        :param done_on_out_of_bound:
        :param kwargs:
        '''
        # todo: something useless now, but will be useful later
        self.state = None
        self.obs_goal_horizon = obs_goal_horizon
        self.obs_wrap_angle = obs_wrap_angle
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.rew_exponential = rew_exponential
        self.done_on_out_of_bound = done_on_out_of_bound
        # todo: super class!!
        super().__init__(**kwargs)

        # create a PyBullet client connection
        self.PYB_CLIENT = -1
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)
        # disable urdf caching for randomization via reload urdf
        p.setPhysicsEngineParameter(enableFileCaching=1)

        # set gui and rendering size
        self.RENDER_HEIGHT = int(200)
        self.RENDER_WIDTH = int(320)

        # set the init state
        # (x, x_dot, theta, theta_dot)
        self.nState = 4
        self.nControl = 1
        if init_state is None:
            self.INIT_X, self.INIT_THETA, self.INIT_X_DOT, self.INIT_THETA_DOT = np.zeros(self.nState)
        elif isinstance(init_state, np.ndarray) and len(init_state) == self.nState:
            self.INIT_X, self.INIT_THETA, self.INIT_X_DOT, self.INIT_THETA_DOT = init_state
        else:
            raise ValueError('[ERROR] in CartPole.__init__(), init_state, type: {}, size: {}'.format(type(init_state),
                                                                                                     len(init_state)))

        # get physical properties from URDF (as default parameters)
        self.GRAVITY = 9.81
        EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS = self._parse_urdf_parameters(self.URDF_PATH)
        if inertial_prop is None:
            self.EFFECTIVE_POLE_LENGTH, self.POLE_MASS, self.CART_MASS = EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS
        elif isinstance(inertial_prop, dict):
            self.EFFECTIVE_POLE_LENGTH = inertial_prop.get('pole_length', EFFECTIVE_POLE_LENGTH)
            self.POLE_MASS = inertial_prop.get('pole_mass', POLE_MASS)
            self.CART_MASS = inertial_prop.get('cart_mass', CART_MASS)
        else:
            raise ValueError(
                '[ERROR] in CartPole.__init__(), inertial_prop, type: {}, size: {}'.format(type(inertial_prop),
                                                                                           len(inertial_prop)))

        # Create X_GOAL and U_GOAL references for the assigned task.
        self.U_GOAL = np.zeros(1)
        if self.Task == Task.STABILIZATION:
            self.X_GOAL = np.array([1, 0, 0, 0])
        else:
            raise ValueError('[ERROR] in CartPole.__init__(), TASK TYPE')

        self._setup_symbolic()

    def reset(self, seed=None):
        super().before_reset(seed=seed)
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(enableRealTimeSimulation=0, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.PYB_CLIENT)
        # Choose randomized or deterministic inertial properties.
        prop_values = {'pole_length': self.EFFECTIVE_POLE_LENGTH, 'cart_mass': self.CART_MASS,
                       'pole_mass': self.POLE_MASS}
        # todo: random model here
        self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH = prop_values['pole_length']
        self.OVERRIDDEN_CART_MASS = prop_values['cart_mass']
        self.OVERRIDDEN_POLE_MASS = prop_values['pole_mass']
        OVERRIDDEN_POLE_INERTIA = (1 / 12) * self.OVERRIDDEN_POLE_MASS * (
                2 * self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH) ** 2
        override_urdf_tree = self._create_urdf(self.URDF_PATH, length=self.OVERRIDDEN_EFFECTIVE_POLE_LENGTH,
                                               inertia=OVERRIDDEN_POLE_INERTIA)
        self.override_path = os.path.join(self.output_dir, f'pid-{os.getpid()}_id-{self.idx}_cartpole.urdf')
        override_urdf_tree.write(self.override_path)

        self.CARTPOLE_ID = p.loadURDF(
            self.URDF_PATH,
            basePosition=[0, 0, 0],
            physicsClientId=self.PYB_CLIENT)
        # Remove cache file after loading it into PyBullet.
        os.remove(self.override_path)
        # Cartpole setting: force control
        for i in [-1, 0, 1]:  # Slider, cart, and pole.
            p.changeDynamics(self.CARTPOLE_ID, linkIndex=i, linearDamping=0, angularDamping=0,
                             physicsClientId=self.PYB_CLIENT)
        for i in [0, 1]:  # Slider-to-cart and cart-to-pole joints.
            p.setJointMotorControl2(self.CARTPOLE_ID, jointIndex=i, controlMode=p.VELOCITY_CONTROL, force=0,
                                    physicsClientId=self.PYB_CLIENT)
        # override inertial properties
        p.changeDynamics(
            self.CARTPOLE_ID,
            linkIndex=0,  # Cart.
            mass=self.OVERRIDDEN_CART_MASS,
            physicsClientId=self.PYB_CLIENT)
        # TODO: HOW TO KNOW POLE CENTER OF MASS???
        p.changeDynamics(
            self.CARTPOLE_ID,
            linkIndex=1,  # Pole.
            mass=self.OVERRIDDEN_POLE_MASS,
            physicsClientId=self.PYB_CLIENT)
        # Randomize initial state.
        init_values = {'init_x': self.INIT_X, 'init_x_dot': self.INIT_X_DOT, 'init_theta': self.INIT_THETA,
                       'init_theta_dot': self.INIT_THETA_DOT}
        if self.RANDOMIZED_INIT:
            init_values = self._randomize_values_by_info(init_values, self.INIT_STATE_RAND_INFO)
        OVERRIDDEN_INIT_X = init_values['init_x']
        OVERRIDDEN_INIT_X_DOT = init_values['init_x_dot']
        OVERRIDDEN_INIT_THETA = init_values['init_theta']
        OVERRIDDEN_INIT_THETA_DOT = init_values['init_theta_dot']

        p.resetJointState(
            self.CARTPOLE_ID,
            jointIndex=0,  # Slider-to-cart joint.
            targetValue=OVERRIDDEN_INIT_X,
            targetVelocity=OVERRIDDEN_INIT_X_DOT,
            physicsClientId=self.PYB_CLIENT)
        p.resetJointState(
            self.CARTPOLE_ID,
            jointIndex=1,  # Cart-to-pole joints.
            targetValue=OVERRIDDEN_INIT_THETA,
            targetVelocity=OVERRIDDEN_INIT_THETA_DOT,
            physicsClientId=self.PYB_CLIENT)

        return self.get_state()

    def step(self, action):
        """

        :param action (ndarray): the action applied to the environments for teh step
        :return:
            obs (ndarray): the observation (y) of the environment after the step
            reward (float): the scalar reward/cost of the step
            done (bool): whether the conditions for the end of an episode are met in the step
            info (dict): a dictionary with information about the constraints evaluations and violations
        """
        # force = super().before_step(action)  # return processed scalar action

        self._apply_force(action)

        self.state = self.get_state()

        return self.state, None, None, None

    def _apply_force(self, force):
        """
        apply force to cartpole, the pybullet simulation is stepped by PYB_FREQ/CTRL_FREQ times

        :param force:
        :return:
        """
        # todo: determinate disturbance
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # todo: add disturbance force here, using applyExternalForce in PyBullet

            # apply force to cartpole
            p.setJointMotorControl2(self.CARTPOLE_ID,
                                    jointIndex=0,  # slider-to-cart joint
                                    controlMode=p.TORQUE_CONTROL,
                                    force=force,
                                    physicsClientId=self.PYB_CLIENT)
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)

    @property
    def get_id(self):
        return self.id

    def get_state(self):
        # [x, theta, x_dot, theta_dot]
        x = p.getJointState(self.CARTPOLE_ID, jointIndex=0, physicsClientId=self.PYB_CLIENT)[0]
        x_dot = p.getJointState(self.CARTPOLE_ID, jointIndex=0, physicsClientId=self.PYB_CLIENT)[1]
        theta = p.getJointState(self.CARTPOLE_ID, jointIndex=1, physicsClientId=self.PYB_CLIENT)[0]
        theta_dot = p.getJointState(self.CARTPOLE_ID, jointIndex=1, physicsClientId=self.PYB_CLIENT)[1]
        theta = utils.normalize_angle(theta)
        state = np.hstack((x, theta, x_dot, theta_dot))

        self.state = np.array(state)
        return self.state

    def execute(self, force):
        p.setJointMotorControl2(self.id, 0, p.TORQUE_CONTROL, force=force)
        p.stepSimulation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _parse_urdf_parameters(self, file_name):
        '''Parses an URDF file for the robot's properties.

        Args:
            file_name (str, optional): The .urdf file from which the properties should be pased.

        Returns:
            EFFECTIVE_POLE_LENGTH (float): The effective pole length.
            POLE_MASS (float): The pole mass.
            CART_MASS (float): The cart mass.
        '''
        URDF_TREE = (etxml.parse(file_name)).getroot()
        EFFECTIVE_POLE_LENGTH = 0.5 * float(
            URDF_TREE[3][0][0][0].attrib['size'].split(' ')[-1])  # Note: HALF length of pole.
        POLE_MASS = float(URDF_TREE[3][1][1].attrib['value'])
        CART_MASS = float(URDF_TREE[1][2][0].attrib['value'])
        return EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS

    def _create_urdf(self, file_name, length=None, inertia=None):
        '''For domain randomization.

        Args:
            file_name (str): Path to the base URDF with attributes to modify.
            length (float): Overriden effective pole length.
            inertia (float): Pole inertia (symmetric, Ixx & Iyy).

        Returns:
            tree (obj): xml tree object.
        '''
        tree = etxml.parse(file_name)
        root = tree.getroot()
        # Overwrite pod length.
        if length is not None:
            # Pole visual geometry box.
            out = root[3][0][0][0].attrib['size']
            out = ' '.join(out.split(' ')[:-1] + [str(2 * length)])
            root[3][0][0][0].attrib['size'] = out
            # Pole visual origin.
            out = root[3][0][1].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][0][1].attrib['xyz'] = out
            # Pole inertial origin.
            out = root[3][1][0].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][1][0].attrib['xyz'] = out
            # Pole inertia.
            root[3][1][2].attrib['ixx'] = str(inertia)
            root[3][1][2].attrib['iyy'] = str(inertia)
            root[3][1][2].attrib['izz'] = str(0.0)
            # Pole collision geometry box.
            out = root[3][2][0][0].attrib['size']
            out = ' '.join(out.split(' ')[:-1] + [str(2 * length)])
            root[3][2][0][0].attrib['size'] = out
            # Pole collision origin.
            out = root[3][2][1].attrib['xyz']
            out = ' '.join(out.split(' ')[:-1] + [str(length)])
            root[3][2][1].attrib['xyz'] = out
        return tree

    def _randomize_values_by_info(self,
                                  original_values,
                                  randomization_info
                                  ):
        '''Randomizes a list of values according to desired distributions.

        Args:
            original_values (dict): A dict of orginal values.
            randomization_info (dict): A dictionary containing information about the distributions
                                       used to randomize original_values.

        Returns:
            randomized_values (dict): A dict of randomized values.
        '''

        # Start from a copy of the original values.
        randomized_values = copy.deepcopy(original_values)
        # Copy the info dict to parse it with 'pop'.
        rand_info_copy = copy.deepcopy(randomization_info)
        # Randomized and replace values for which randomization info are given.
        for key in original_values:
            if key in rand_info_copy:
                # Get distribution removing it from info dict.
                distrib = getattr(self.np_random,
                                  rand_info_copy[key].pop('distrib'))
                # Pop positional args.
                d_args = rand_info_copy[key].pop('args', [])
                # Keyword args are just anything left.
                d_kwargs = rand_info_copy[key]
                # Randomize (adding to the original values).
                randomized_values[key] += distrib(*d_args, **d_kwargs)
        return randomized_values

    def _setup_symbolic(self, prior_prob={}):
        l = prior_prob.get("pole_length", self.EFFECTIVE_POLE_LENGTH)
        m = prior_prob.get("pole_mass", self.POLE_MASS)  # m2
        M = prior_prob.get("cart_mass", self.CART_MASS)  # m1
        Mm, ml = m + M, m * l
        g = self.GRAVITY
        dt = self.CTRL_TIMESTEP
        # # Input variable
        # q1 = cs.MX.sym('q1')  # x
        # q2 = cs.MX.sym('q2')  # theta
        # dq1 = cs.MX.sym('dq1')  # x_dot
        # dq2 = cs.MX.sym('dq2')  # theta_dot
        # q = cs.vertcat(q1, q2)
        # dq = cs.vertcat(dq1, dq2)
        # X = cs.vertcat(q, dq)
        # U = cs.MX.sym('U')
        # nx = self.nState
        # nu = self.nControl
        # # todo: it is different from safe-control-gym, check whether correct?
        # ddq1 = (ml * cs.sin(q2) * dq2 ** 2 + U + m * g * cs.cos(q2) * cs.sin(q2)) / (M + m * (1 - cs.cos(q2) ** 2))
        # ddq2 = -(ml * cs.cos(q2) * cs.sin(q2) * dq2 ** 2 + U * cs.cos(q2) + Mm * g * cs.sin(q2)) / (
        #         l * M + ml * (1 - cs.cos(q2) ** 2))
        # ddq = cs.vertcat(ddq1, ddq2)
        # X_dot = cs.vertcat(dq, ddq)
        # # observation
        # Y = cs.vertcat(q, dq)

        # todo: this is safe control gym
        # Input variables.
        x = cs.MX.sym('x')
        x_dot = cs.MX.sym('x_dot')
        theta = cs.MX.sym('theta')
        theta_dot = cs.MX.sym('theta_dot')
        X = cs.vertcat(x, theta, x_dot, theta_dot)
        U = cs.MX.sym('U')
        nx = 4
        nu = 1
        # Dynamics.
        temp_factor = (U + ml * theta_dot ** 2 * cs.sin(theta)) / Mm
        theta_dot_dot = (
                    (g * cs.sin(theta) - cs.cos(theta) * temp_factor) / (l * (4.0 / 3.0 - m * cs.cos(theta) ** 2 / Mm)))
        X_dot = cs.vertcat(x_dot, theta_dot, temp_factor - ml * theta_dot_dot * cs.cos(theta) / Mm, theta_dot_dot)
        # Observation.
        Y = cs.vertcat(x, theta, x_dot, theta_dot)
        # define cost (quadratic form)
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T * R * (U - Ur)
        # define dyn and cost dictionaries
        first_dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
        cost = {'cost_func': cost_func, 'vars': {'X': X, 'Xr': Xr, 'U': U, 'Ur': Ur, 'Q': Q, 'R': R}}
        # parameters
        params = {
            # prior inertial properties
            'pole_length': l,
            'pole_mass': m,
            'cart_mass': M,
            # equilibrium point for linearization
            'X_EQ': np.array([1,0,0,0]),
            'U_EQ': np.zeros(self.nControl)  # np.atleast_2d(self.U_GOAL)[0, :],
        }
        self.symbolic = FirstOrderModel(dynamics=first_dynamics, dt=self.CTRL_TIMESTEP,
                                        cost=cost, params=params)

    def _preprocess_control(self, action):
        action = self._denormalize_action(action)

        self.current_physical_action = action

        # todo: add action disturbances here
        self.current_noisy_physical_action = action

        # clip action
        force = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])  # nonempty
        self.current_clipped_action = force

        return force[0]

    def _set_action_space(self):
        self.action_limit = 10
        self.physical_action_bounds = (-1 * np.atleast_1d(self.action_limit), -1 * np.atleast_1d(self.action_limit))
        self.action_threshold = 1
        self.action_space = spaces.Box(low=-self.action_threshold, high=self.action_threshold, shape=(1,))

        # define action/input labels and units
        self.ACTION_LABELS = ['U']
        self.ACTION_UNITS = ['N']

    def _denormalize_action(self, action):
        """ converts a normalized action into a physical action, only need in RL-based action
        :param action (ndarray):
        :return: action (ndarray):
        """
        return action

from functools import partial

if __name__ == '__main__':
    key_word = {'gui': False}
    env_func = partial(CartPole, **key_word)
    q_lqr = [1]
    r_lqr = [0.1]
    lqr_controller = LQR(env_func=env_func, q_lqr=q_lqr, r_lqr=r_lqr, discrete_dynamics=True)
    print("start")
    cart_pole = CartPole(gui=True)
    cart_pole.reset()
    while 1:
        current_state = cart_pole.get_state()
        action = lqr_controller.select_action(current_state)
        print("action: {}".format(action))
        cart_pole.step(action)
        print(cart_pole.get_state())
        time.sleep(0.1)
    print("cart pole dyn func: {}".format(cart_pole.symbolic.fc_func))
    while 1:
        pass
