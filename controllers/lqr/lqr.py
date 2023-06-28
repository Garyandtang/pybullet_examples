"""
Linear Quadratic Regulator (LQR)
"""
from controllers.base_controller import BaseController
from controllers.lqr.lqr_utils import *
from utils.enum_class import Task


class LQR(BaseController):
    def __init__(self,
                 env_func,
                 q_lqr: list = None,
                 r_lqr: list = None,
                 discrete_dynamics: bool = True,
                 **kwargs):
        """
        LQR controller
        :param env_func:
        :param q_lqr:
        :param r_lqr:
        :param discrete_dynamics:
        :param kwargs:
        """
        super().__init__(env_func, **kwargs)
        # create an env with model
        self.env = env_func()
        self.model = self.get_prior(self.env)
        self.Q = get_cost_weight_matrix(q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(r_lqr, self.model.nu)
        self.discrete_dynamics = discrete_dynamics

        self.gain = compute_lqr_gain(self.model, self.model.X_EQ, self.model.U_EQ,
                                     self.Q, self.R, self.discrete_dynamics)

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def select_action(self, obs, info=None):
        step = self.extract_step(info)

        if self.env.Task == Task.STABILIZATION:
            return -self.gain @ (obs - self.env.X_GOAL) + self.model.U_EQ
        elif self.env.TASK == Task.TRAJ_TRACKING:
            return -self.gain @ (obs - self.env.X_GOAL[step]) + self.model.U_EQ
