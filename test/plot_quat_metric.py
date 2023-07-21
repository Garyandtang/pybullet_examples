import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from liecasadi import SO3, SO3Tangent

# matplotlib.use('TKAgg')#加上这行代码即可，agg是一个没有图形显示界面的终端，常用的有图形界面显示的终端有TkAgg等

def set_theta_metric():
    yaw = ca.MX.sym('yaw', 1)
    rpy = ca.vertcat(0, 0, yaw)
    so3 = SO3.from_euler(rpy)
    yaw_tar = ca.MX.sym('yaw_tar', 1)
    rpy_tar = ca.vertcat(0, 0, yaw_tar)
    so3_tar = SO3.from_euler(rpy_tar)

    # function calculating the difference of two theta using quaternion
    diff_in_quat = 1 - ca.power(ca.mtimes(ca.transpose(so3.as_quat().coeffs()), so3_tar.as_quat().coeffs()), 2)
    func_diff_in_quat = ca.Function('diff_in_quat', [yaw, yaw_tar], [diff_in_quat])

    # function calculating the difference of two theta using theta
    diff_in_theta = 1 - ca.cos(yaw - yaw_tar)
    func_diff_in_theta = ca.Function('diff_in_theta', [yaw, yaw_tar], [diff_in_theta])

    return func_diff_in_quat, func_diff_in_theta


if __name__ == '__main__':
    yaw_vec = np.arange(-3*np.pi, 3*np.pi, 0.1)
    yaw_tar = 0
    func_diff_in_quat, func_diff_in_theta = set_theta_metric()

    # record difference vec
    diff_in_quat_vec = []
    diff_in_theta_vec = []
    for yaw in yaw_vec:
        diff_in_quat = func_diff_in_quat(yaw, yaw_tar).full()[0][0]
        diff_in_theta = func_diff_in_theta(yaw, yaw_tar).full()[0][0]
        diff_in_quat_vec.append(diff_in_quat)
        diff_in_theta_vec.append(diff_in_theta)

    # plot
    # vec
    plt.plot(yaw_vec, np.array(diff_in_quat_vec), label='diff_in_quat')
    plt.plot(yaw_vec, np.array(diff_in_theta_vec), label='diff_in_theta')
    plt.xlabel('yaw')
    plt.ylabel('difference')
    plt.title('quat: 1 - <q1, q2>^2, theta: 1 - cos(theta)')
    plt.legend()
    plt.show()

    # save img
    plt.savefig('quat_metric.png')
