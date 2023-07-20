import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from liecasadi import SO3, SO3Tangent
# import manifpy

def set_quat_diff_func():
    # convert yaw to SO3 representation
    yaw = cs.MX.sym('yaw', 1)
    rpy = cs.vertcat(0, 0, yaw)
    so3 = SO3.from_euler(rpy)
    rpy_tar = cs.vertcat(0, 0, 0)
    so3_tar = SO3.from_euler(rpy_tar)

    # d(q1, q2) = 1 - <q1, q2>^2
    quat_diff = 1 - cs.power(cs.mtimes(cs.transpose(so3.as_quat().coeffs()), so3_tar.as_quat().coeffs()), 2)

    quat_diff_func = cs.Function('quat_diff_func', [yaw], [quat_diff])

    return quat_diff_func

if __name__ == '__main__':
    quat_diff_func = set_quat_diff_func()
    yaw_vec = np.arange(-3*np.pi, 3*np.pi, 0.1)
    quat_diff_vec = []
    for yaw in yaw_vec:
        diff = quat_diff_func(yaw).full()[0][0]
        quat_diff_vec.append(diff)

    plt.plot(yaw_vec, quat_diff_vec)
    # set plot x, y label
    plt.xlabel('yaw (rad)')
    plt.title('orientation difference between yaw and zero yaw')

    plt.show()
    plt.savefig('quat_diff.png')
