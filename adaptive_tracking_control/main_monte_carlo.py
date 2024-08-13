from data_driven_FBC import *
import os
import time
from planner.ref_traj_generator import TrajGenerator
from matplotlib import ticker
def MonteCarlo():
    # get reference trajectory
    ref_sysm = LTI()
    traj_config = {'type': TrajType.CIRCLE,
                     'param': {'start_state': np.zeros((3,)),
                              'linear_vel': ref_sysm.v,
                              'angular_vel': ref_sysm.w,
                              'nTraj': 299,
                              'dt': 0.02}}
    traj_gen = TrajGenerator(traj_config)
    ref_state, ref_control, dt = traj_gen.get_traj()
    ref_x = ref_state[0, :]
    ref_y = ref_state[1, :]
    totalSim = 50
    iteration = 5
    y_max = 0.8
    y_min = -0.3
    K0_container = np.zeros((totalSim, iteration + 1))
    K1_container = np.zeros((totalSim, iteration + 1))
    K2_container = np.zeros((totalSim, iteration + 1))
    K3_container = np.zeros((totalSim, iteration + 1))
    K4_container = np.zeros((totalSim, iteration + 1))
    K5_container = np.zeros((totalSim, iteration + 1))
    k0_container = np.zeros((totalSim, iteration + 1))
    k1_container = np.zeros((totalSim, iteration + 1))
    r_container = np.zeros((totalSim, iteration+1))
    l_container = np.zeros((totalSim, iteration+1))
    nTraj = 300
    x_trained_container = np.zeros((totalSim, nTraj-1))
    y_trained_container = np.zeros((totalSim, nTraj-1))

    x_init_container = np.zeros((totalSim, nTraj - 1))
    y_init_container = np.zeros((totalSim, nTraj - 1))

    init_x_error = np.zeros((totalSim, ))
    init_y_error = np.zeros((totalSim, ))
    learned_x_error = np.zeros((totalSim, ))
    learned_y_error = np.zeros((totalSim, ))
    totalFail = 0
    i = 0
    avg_learning_time = 0
    while i < totalSim:
        lti = LTI()
        x_init_container[i, :], y_init_container[i, :], _, _ = evaluation(lti, nTraj)
        # calculate the initial error
        init_x_error[i] = np.linalg.norm(x_init_container[i, :] - ref_x)
        init_y_error[i] = np.linalg.norm(y_init_container[i, :] - ref_y)
        r_container[i, 0], l_container[i, 0] = calculate_r_l(lti.B, lti.dt)
        init_K = lti.K_ini
        init_k = lti.k_ini
        K0_container[i, 0] = init_K[0, 0]
        K1_container[i, 0] = init_K[0, 1]
        K2_container[i, 0] = init_K[0, 2]
        K3_container[i, 0] = init_K[1, 0]
        K4_container[i, 0] = init_K[1, 1]
        K5_container[i, 0] = init_K[1, 2]
        k0_container[i, 0] = init_k[0]
        k1_container[i, 0] = init_k[1]
        optimal_K = lti.K_ground_truth
        optimal_k = lti.k_ground_truth
        for j in range(iteration):
            start_time = time.time()
            K, k, B, successful = learning(lti)
            learning_time = time.time() - start_time
            avg_learning_time += learning_time
            print("learning time: ", learning_time)
            if not successful:
                print("failed to learn")
                totalFail += 1
                i = i - 1
                break
            r, l = calculate_r_l(B, lti.dt)
            K0_container[i, j + 1] = K[0, 0]  # - optimal_K[0, 0]
            K1_container[i, j + 1] = K[0, 1]  # - optimal_K[0, 1]
            K2_container[i, j + 1] = K[0, 2]  # - optimal_K[0, 2]
            K3_container[i, j + 1] = K[1, 0]  # - optimal_K[1, 0]
            K4_container[i, j + 1] = K[1, 1]  # - optimal_K[1, 1]
            K5_container[i, j + 1] = K[1, 2]  # - optimal_K[1, 2]
            k0_container[i, j + 1] = k[0]  # - optimal_k[0]
            k1_container[i, j + 1] = k[1]  # - optimal_k[1]
            r_container[i, j+1] = r
            l_container[i, j+1] = l

        x_trained_container[i, :], y_trained_container[i, :], _, _ = evaluation(lti, nTraj)
        # calculate the learned error
        learned_x_error[i] = np.linalg.norm(x_trained_container[i, :] - ref_x)
        learned_y_error[i] = np.linalg.norm(y_trained_container[i, :] - ref_y)
        i = i + 1

    # calculate the average learning time
    avg_learning_time = avg_learning_time / (totalSim * iteration)
    print("avg learning time: ", avg_learning_time)
    font_size = 16
    line_width = 2
    # dirpath
    dirpath = os.getcwd()
    data_path = os.path.join(dirpath, "data", "mote_carlo")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # box plot initial r and final r
    r_init_container = r_container[:, 0]
    r_final_container = r_container[:, -1]
    plt.figure()
    # box plot with color
    datas = [r_init_container, r_final_container]
    colors = [[0.4940, 0.1840, 0.5560], [0.9290, 0.6940, 0.1250]]
    bplot = plt.boxplot(datas, showfliers=False, patch_artist=True, boxprops={'facecolor': 'none', 'alpha': 0.5})
    plt.xticks([1, 2], ['initial $r$', 'learned $r$'], fontsize=font_size)
    plt.ylabel('length ($m$)', fontsize=font_size)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    # save bplot
    plt.savefig(os.path.join(data_path, "r_boxplot.jpg"))
    plt.show()
    # box plot initial l and final l
    l_init_container = l_container[:, 0]
    l_final_container = l_container[:, -1]
    plt.figure()
    colors = [[0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250]]
    bplot = plt.boxplot([l_init_container, l_final_container],showfliers=False, patch_artist=True, boxprops={'facecolor': 'none', 'alpha': 0.5})
    plt.xticks([1, 2], ['initial $l$', 'learned $l$'], fontsize=font_size)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel('length ($m$)', fontsize=font_size)

    plt.savefig(os.path.join(data_path, "l_boxplot.jpg"))
    plt.show()

    # box plot initial x error and final x error
    x_init_error_container = init_x_error
    x_final_error_container = learned_x_error
    plt.figure()
    colors = [[0.9290, 0.6940, 0.1250], [0.9290, 0.6940, 0.1250]]
    bplot = plt.boxplot([x_init_error_container, x_final_error_container],showfliers=False, patch_artist=True, boxprops={'facecolor': 'none', 'alpha': 0.5})
    plt.xticks([1, 2], ['initial $x^{p}$ error', 'learned $x^{p}$ error'], fontsize=font_size)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel('Mean Square Error ($m$)', fontsize=font_size)

    plt.savefig(os.path.join(data_path, "x_error_boxplot.jpg"))
    plt.show()

    # box plot initial y error and final y error
    y_init_error_container = init_y_error
    y_final_error_container = learned_y_error
    plt.figure()
    colors = [[0, 0.4470, 0.7410], [0.9290, 0.6940, 0.1250]]
    bplot = plt.boxplot([y_init_error_container, y_final_error_container],showfliers=False, patch_artist=True, boxprops={'facecolor': 'none', 'alpha': 0.5})
    plt.xticks([1, 2], ['initial $y^{p}$ error', 'learned $y^{p}$ error'], fontsize=font_size)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.ylabel('Mean Square Error ($m$)', fontsize=font_size)
    plt.savefig(os.path.join(data_path, "y_error_boxplot.jpg"))
    plt.show()


    # plot init x and y in the same figure
    plt.figure()
    plt.grid(True)


    plt.xticks(fontsize=font_size-2)
    plt.yticks(fontsize=font_size-2)
    plt.plot(x_init_container.T, y_init_container.T, linewidth=line_width)
    # plt.tight_layout()
    plt.xlabel("$x~(m)$", fontsize=font_size)
    plt.ylabel("$y~(m)$", fontsize=font_size)
    name = "init_trajectory.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()

    # plot trained x and y in the same figure
    plt.figure()
    plt.grid(True)
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
    # show grid
    plt.grid(True)
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
    plt.grid(True)
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
    np.save('data/r_container.npy', r_container)
    np.save('data/l_container.npy', l_container)
    np.save('data/x_init_container.npy', x_init_container)
    np.save('data/y_init_container.npy', y_init_container)
    np.save('data/x_trained_container.npy', x_trained_container)
    np.save('data/y_trained_container.npy', y_trained_container)



if __name__ == '__main__':
    MonteCarlo()
