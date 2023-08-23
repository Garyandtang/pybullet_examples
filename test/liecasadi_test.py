import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from liecasadi import SO3, SO3Tangent
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent

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
    # 3 element random int numpy vector
    x = np.random.randint(0, 10, 3)
    se2 = SE2Tangent(x)
    print(se2.coeffs())
    xx = se2*2

    t = 3
    a = (t*xx).exp()
    b = xx.exp() * xx.exp() * xx.exp()
    print(a.coeffs())
    print(b.coeffs())


