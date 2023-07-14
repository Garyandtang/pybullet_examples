from controllers.lqr.lqr_utils import *

class MPC():
    def __init__(self,
                 env_func,
                 q_lqr: list = None,
                 r_lqr: list = None,
                discrete_dynamics: bool = True,
                **kwargs):
        self.env = env_func()
        self.model = self.get_prior(self.env)
        self.Q = get_cost_weight_matrix(q_lqr, self.model.nx)
        self.R = get_cost_weight_matrix(r_lqr, self.model.nu)
        self.discrete_dynamics = discrete_dynamics


    def select_action(self, obs, info=None):
        raise NotImplementedError

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def get_prior(self, env):
        raise NotImplementedError

    def setup_optimizer(self):
        raise NotImplementedError

    def compute_init_guess(self):
        raise NotImplementedError

