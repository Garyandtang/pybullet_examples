from data_driven_FBC import *
import os

def get_data():
    lti = LTI()
    nTraj = 300
    _, _, error_container, control_container = evaluation(lti, nTraj, learning=True)

    # dirpath
    dirpath = os.getcwd()
    data_path = os.path.join(dirpath, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    font_size = 17
    line_width = 2
    # plot error[0]
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(error_container[0, :], linewidth=line_width)
    plt.xlabel("k", fontsize=font_size)
    plt.title("$x[0]~(m)$", fontsize=font_size)
    name = "error_x.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()

    # plot error[1]
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(error_container[1, :], linewidth=line_width)
    plt.xlabel("k", fontsize=font_size)
    plt.title("$x[1]~(m)$", fontsize=font_size)
    name = "error_y.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()

    # plot error[2]
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(error_container[2, :], linewidth=line_width)
    plt.xlabel("k", fontsize=font_size)
    plt.title("$x[2]~(rad)$", fontsize=font_size)
    name = "error_theta.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()


    # plot control[0] and control[1]
    plt.figure()
    plt.grid(True)
    plt.xticks(fontsize=font_size - 4)
    plt.yticks(fontsize=font_size - 4)
    plt.plot(control_container[0, :], linewidth=line_width)
    plt.plot(control_container[1, :], linewidth=line_width)
    plt.xlabel("k", fontsize=font_size)
    plt.title("$u~(rad/s)$", fontsize=font_size)
    plt.legend(['$u[0]$', '$u[1]$'], fontsize=font_size - 2)
    name = "control.jpg"
    plt.savefig(os.path.join(data_path, name))
    plt.show()







if __name__ == '__main__':
    get_data()