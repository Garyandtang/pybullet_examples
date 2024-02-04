from data_driven_FBC import *
import os
from matplotlib import ticker
def MonteCarlo():
    totalSim = 50
    iteration = 5
    y_max = 0.8
    y_min = -0.3
    K0_container = np.zeros((totalSim, iteration))
    K1_container = np.zeros((totalSim, iteration))
    K2_container = np.zeros((totalSim, iteration))
    K3_container = np.zeros((totalSim, iteration))
    K4_container = np.zeros((totalSim, iteration))
    K5_container = np.zeros((totalSim, iteration))
    k0_container = np.zeros((totalSim, iteration))
    k1_container = np.zeros((totalSim, iteration))
    r_container = np.zeros((totalSim, iteration+1))
    l_container = np.zeros((totalSim, iteration+1))
    nTraj = 1700
    x_trained_container = np.zeros((totalSim, nTraj-1))
    y_trained_container = np.zeros((totalSim, nTraj-1))

    x_init_container = np.zeros((totalSim, nTraj - 1))
    y_init_container = np.zeros((totalSim, nTraj - 1))

    totalFail = 0
    for i in range(totalSim):
        lti = LTI()
        x_init_container[i, :], y_init_container[i, :] = evaluation(lti, nTraj)
        r_container[i, 0], l_container[i, 0] = calculate_r_l(lti.B, lti.dt)
        optimal_K = lti.K_optimal
        optimal_k = lti.k_optimal
        for j in range(iteration):
            K, k, B, successful = learning(lti)
            if not successful:
                print("failed to learn")
                totalFail += 1
                i = i - 1
                continue
            r, l = calculate_r_l(B, lti.dt)
            K0_container[i, j] = K[0, 0] - optimal_K[0, 0]
            K1_container[i, j] = K[0, 1] - optimal_K[0, 1]
            K2_container[i, j] = K[0, 2] - optimal_K[0, 2]
            K3_container[i, j] = K[1, 0] - optimal_K[1, 0]
            K4_container[i, j] = K[1, 1] - optimal_K[1, 1]
            K5_container[i, j] = K[1, 2] - optimal_K[1, 2]
            k0_container[i, j] = k[0] - optimal_k[0]
            k1_container[i, j] = k[1] - optimal_k[1]
            r_container[i, j+1] = r
            l_container[i, j+1] = l

        x_trained_container[i, :], y_trained_container[i, :] = evaluation(lti, nTraj)

    font_size = 12
    line_width = 2
    # dirpath
    dirpath = os.getcwd()
    data_path = os.path.join(dirpath, "data")
    # plot init x and y in the same figure
    plt.figure()



    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.plot(x_init_container.T, y_init_container.T, linewidth=line_width)
    # plt.tight_layout()
    plt.xlabel("$x~(m)$",fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)
    name = "init_trajectory.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()

    # plot trained x and y in the same figure
    plt.figure()
    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.plot(x_trained_container.T, y_trained_container.T, linewidth=line_width)
    plt.xlabel("$x~(m)$", fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)
    name = "trained_trajectory.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()

    # plot r
    x = np.arange(0, iteration+1, 1)
    plt.figure()
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)
    plt.plot(x, r_container.T, linewidth=line_width)
    plt.xlabel("iteration", fontsize=font_size)
    plt.ylabel("$r~(m)$", fontsize=font_size)
    name = "trained_r.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()

    # plot l
    plt.figure()
    plt.xticks(fontsize=font_size - 2)
    plt.yticks(fontsize=font_size - 2)

    plt.plot(x, l_container.T, linewidth=line_width)
    plt.xlabel("iteration", fontsize=font_size)
    plt.ylabel("$l~(m)$", fontsize=font_size)
    name = "trained_l.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()

    # calculate the mean and std of last r
    r_mean = np.mean(r_container[:, -1])
    r_std = np.std(r_container[:, -1])
    print("r_mean: ", r_mean)
    print("r_std: ", r_std)

    # calculate the mean and std of last l
    l_mean = np.mean(l_container[:, -1])
    l_std = np.std(l_container[:, -1])
    print("l_mean: ", l_mean)
    print("l_std: ", l_std)

    print("failed to learn: ", totalFail)
    # save data
    np.save('data/good/r_container.npy', r_container)
    np.save('data/good/l_container.npy', l_container)
    np.save('data/good/x_init_container.npy', x_init_container)
    np.save('data/good/y_init_container.npy', y_init_container)
    np.save('data/good/x_trained_container.npy', x_trained_container)
    np.save('data/good/y_trained_container.npy', y_trained_container)


if __name__ == '__main__':
    MonteCarlo()
