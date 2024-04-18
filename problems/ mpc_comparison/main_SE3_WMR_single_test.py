import numpy as np
from utils.enum_class import TrajType, ControllerType, EnvType
from controller.nonlinear_mpc import NonlinearMPC
from controller.feedback_linearization import FBLinearizationController
from controller.SE3_mpc import SE3MPC
from controller.geometric_mpc import GeometricMPC
from planner.ref_traj_generator import TrajGenerator
from problems.unicycle_traj_tracking.monte_carlo_test_turtlebot import simulation
from controller.se3_traj_generator import SE3TrajGenerator
import os
import matplotlib.pyplot as plt
from utils.enum_class import CostType, DynamicsType, ControllerType
from environments.wheeled_mobile_robot.turtlebot.turtlebot import Turtlebot
from environments.wheeled_mobile_robot.scout.scout_mini import ScoutMini
import pybullet as p
def single_tracking_task():
    # set wheel mobile robot
    env_type = EnvType.TURTLEBOT
    # set controller
    controller_type = ControllerType.SE3MPC
    # set init state
    init_state = np.array([0, 0, 0])
    # set reference trajetory
    # traj_config = {'type': TrajType.CIRCLE,
    #                'param': {'start_state': np.array([0, 0, 0]),
    #                          'linear_vel': 0.02,
    #                          'angular_vel': 0.05,
    #                          'nTraj': 1500,
    #                          'dt': 0.02}}
    traj_config = {'type': TrajType.POSE_REGULATION,
              'param': {'end_pos': np.array([1, 1, 0]),
                        'end_euler': np.array([0, 0, np.pi / 2]),
                        'dt': 0.02,
                        'nTraj': 1200}}
    traj_gen = SE3TrajGenerator(traj_config)
    if controller_type == ControllerType.SE3MPC:
        controller = SE3MPC(traj_config)

        # set env and traj
    if env_type == EnvType.TURTLEBOT:
        env = Turtlebot(gui=True, debug=True, init_state=init_state)
    elif env_type == EnvType.SCOUT_MINI:
        env = ScoutMini(gui=True, debug=True, init_state=init_state)
    else:
        raise NotImplementedError
    v_min, v_max, w_min, w_max = env.get_vel_cmd_limit()
    ref_state, ref_control, dt = traj_gen.get_traj()

    # set controller limits
    # controller.set_control_bound(v_min, v_max, w_min, w_max)

    # store simulation traj
    nTraj = ref_state.shape[1]
    store_state = np.zeros((7, nTraj))
    store_control = np.zeros((2, nTraj))
    store_solve_time = np.zeros(nTraj - 1)
    env.draw_ref_traj(ref_state)
    t = 0
    for i in range(nTraj - 1):
        curr_state = env.get_full_state()
        store_state[:, i] = curr_state
        store_control[:, i] = env.get_twist()
        vel_cmd = controller.solve(curr_state, t)
        print("vel_cmd: ", vel_cmd)
        vel_cmd = np.array([vel_cmd[0], vel_cmd[1], vel_cmd[-1]])
        store_solve_time[i] = controller.get_solve_time()
        pos = curr_state[:3]
        quat = curr_state[3:]
        rpy = p.getEulerFromQuaternion(quat)
        print("rpy: ", rpy)
        t += dt
        twist = np.array([vel_cmd[0], 0, vel_cmd[2]])
        env.step(env.twist_to_control(twist))

    store_state[:, -1] = env.get_state()
    store_control[:, -1] = env.get_twist()


def main():
    single_tracking_task()


if __name__ == '__main__':
    main()