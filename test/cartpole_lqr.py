
import time
import casadi as cs
import numpy as np
from functools import partial

from controllers.lqr.lqr import LQR

from envs.cartpole.cartpole import CartPole


if __name__ == '__main__':
    key_word = {'gui': False}
    env_func = partial(CartPole, **key_word)
    q_lqr = [1]
    r_lqr = [0.1]
    lqr_controller = LQR(env_func=env_func, q_lqr=q_lqr, r_lqr=r_lqr, discrete_dynamics=True)
    print("start")
    key_word = {'init_state': np.array([0, np.pi/3, 0, 0]), 'gui': True}
    cart_pole = CartPole(**key_word)
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
