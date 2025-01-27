"""ts.py

"""

import numpy as np
import adaptive_lqr.utils as utils
import logging
import math
import time
from adaptive_control_SE3.adaptive_lqr.adaptive import AdaptiveMethod



class LieAlgebraStrategy(AdaptiveMethod):
    """Adaptive control of our Lie algebra method

    """

    def __init__(self, Q, R, A_star, B_star, sigma_w, rls_lam,
                 reg, tau, actual_error_multiplier):
        super().__init__(Q, R, A_star, B_star, sigma_w, rls_lam)
        self._reg = reg
        self._tau = tau
        self._actual_error_multiplier = actual_error_multiplier
        self._logger = logging.getLogger(__name__)

    def _get_logger(self):
        return self._logger

    def reset(self, rng):
        super().reset(rng)
        self._emp_cov = self._reg * np.eye(self._n + self._p)
        self._last_emp_cov = self._reg * np.eye(self._n + self._p)

    def _design_controller(self, states, inputs, transitions, rng):

        logger = self._get_logger()

        epoch_id = self._epoch_idx + 1 if self._has_primed else 0

        logger.debug("_design_controller(epoch={}): have {} points for regression".format(epoch_id, inputs.shape[0]))

        # do a least squares fit and design based on the nominal
        Anom, Bnom, emp_cov = utils.solve_least_squares(states, inputs, transitions, reg=self._reg)



        return (Anom, Bnom, None)

    def _on_iteration_completion(self):
        # this is called after we take a step
        zt = np.hstack((self._state_history[-1], self._input_history[-1]))
        self._emp_cov += np.outer(zt, zt)

    def _on_epoch_completion(self):
        self._last_emp_cov = np.array(self._emp_cov) # need to make a copy

    def _should_terminate_epoch(self):

        # hack: otherwise in the beginning the epochs are very short
        min_epoch_time = 10
        assert self._tau > min_epoch_time, "make tau larger, or min_epoch_time smaller"

        if self._iteration_within_epoch_idx <= min_epoch_time:
            return False

        # TODO(stephentu): what is the best numerical recipe for this
        # calculation?
        if (np.linalg.det(self._emp_cov) > 2 * np.linalg.det(self._last_emp_cov)) or \
                (self._iteration_within_epoch_idx >= self._tau):
            # condition triggered
            return True
        else:
            # keep going
            return False

    def _get_input(self, state, rng):
        rng = self._get_rng(rng)
        ctrl_input = self._current_K.dot(state)
        return ctrl_input



def _main_se3():
    from adaptive_control_SE3.linear_se3_error_dynamics import LinearSE3ErrorDynamics, evaluation
    lti = LinearSE3ErrorDynamics()
    A = lti.A
    B = lti.B
    Q = lti.Q
    R = lti.R
    K_init = lti.K0
    print("A: ", A)
    print("B: ", B)
    print("Initial I: ", lti.I)
    print("Initial m: ", lti.m)
    #
    init_state_container = evaluation()

    rng = np.random
    env = LieAlgebraStrategy(Q=Q,
                        R=R,
                        A_star=A,
                        B_star=B,
                        sigma_w=0,
                        reg=1e-5,
                        tau=500,
                        actual_error_multiplier=1,
                        rls_lam=None)
    env.reset(rng)
    start_time = time.time()
    env.prime(200, K_init, 5, rng, lti)
    print('prime time:', time.time() - start_time)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    np.set_printoptions(linewidth=200)
    _main_se3()