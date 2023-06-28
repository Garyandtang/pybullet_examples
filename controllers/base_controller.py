"""
Base Controller
"""
from abc import ABC, abstractmethod


class BaseController(ABC):
    def __init__(self,
                 env_func,
                training=True,
                checkpoint_path='temp/model_latest.pt',
                output_path='temp',
                use_gpu=False,
                seed=0,
                **kwargs):
        self.env_func = env_func
        self.training = training
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_path
        self.use_gpu = use_gpu
        self.seed = seed
        self.prior_info = {}
        self.results_dict = {}

        for key, value in kwargs.items():
            self.__dict__[key] = value

        self.reset_result_dict()


    def reset_result_dict(self):
        self.results_dict = {}

    @abstractmethod
    def select_action(self, obs, info=None):
        raise NotImplementedError

    def extract_step(self, info=None):
        if info is not None:
            step = info['current_step']
        else:
            step = 0

        return step

    def learn(self, env=None, **kwargs):
        raise NotImplementedError

    def reset_before_run(self, obs=None, info=None, env=None):
        self.reset_result_dict()

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def get_prior(self, env, prior_info={}):
        if not prior_info:
            prior_info = getattr(self, 'prior_info', {})
        prior_prop = prior_info.get('prior_prop', {})

        # randomize prior prop, similar to randomizing the inertial_prop in BenchmarkEnv
        # this can simulate the estimation errors in the prior model
        randomize_prior_prop = prior_info.get('randomize_prior_prop', False)
        prior_prop_rand_info = prior_info.get('prior_prop_rand_info', {})

        if randomize_prior_prop and prior_prop_rand_info:
            # check keys, this is due to the current implementation of BenchmarkEnv._randomize_values_by_info()
            for k in prior_prop_rand_info:
                assert k in prior_prop, 'A prior param to randomize does not have a base value in prior_prop.'
            prior_prop = env._randomize_values_by_info(prior_prop, prior_prop_rand_info)

        if prior_prop:
            env._setup_symbolic(prior_prop=prior_prop)

        return env.symbolic