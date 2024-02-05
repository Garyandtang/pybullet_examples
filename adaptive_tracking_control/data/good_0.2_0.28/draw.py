import numpy as np
import matplotlib.pyplot as plt
from controller.ref_traj_generator import TrajGenerator
from utils.enum_class import TrajType, ControllerType, LiniearizationType


def main():
    traj_config = {'type': TrajType.CIRCLE,
                   'param': {'start_state': np.zeros((3,)),
                             'linear_vel': 0.02,
                             'angular_vel': 0.2,
                             'nTraj': 1700,
                             'dt': 0.02}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_velocity, dt = traj_gen.get_traj()
    l_container = np.load('l_container.npy')
    r_container = np.load('r_container.npy')
    x_init_container = np.load('x_init_container.npy')
    y_init_container = np.load('y_init_container.npy')
    x_trained_container = np.load('x_trained_container.npy')
    y_trained_container = np.load('y_trained_container.npy')

    # plot init x and y in the same figure
    plt.figure()
    plt.grid(True)
    font_size = 13
    line_width = 2
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.plot(x_init_container.T, y_init_container.T, linewidth=line_width)
    # plt.tight_layout()
    plt.xlabel("$x~(m)$", fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)

    name = "init_trajectory.jpg"
    plt.savefig(name)
    plt.show()

    # plot trained x and y in the same figure
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.plot(x_trained_container.T, y_trained_container.T, linewidth=line_width)
    plt.xlabel("$x~(m)$", fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)
    name = "trained_trajectory.jpg"
    plt.savefig(name)
    plt.show()

    # plot r
    iteration = r_container.shape[1] - 1
    x = np.arange(0, iteration + 1, 1)
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.plot(x, r_container.T, linewidth=line_width)
    plt.xlabel("iteration", fontsize=font_size)
    plt.ylabel("$r~(m)$", fontsize=font_size)
    name = "trained_r.jpg"
    plt.savefig(name)
    plt.show()

    # plot l
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.savefig(name)
    plt.plot(x, l_container.T, linewidth=line_width)
    plt.xlabel("iteration", fontsize=font_size)
    plt.ylabel("$l~(m)$", fontsize=font_size)
    name = "trained_l.jpg"
    plt.savefig(name)
    plt.show()



if __name__ == '__main__':
    main()