import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from liecasadi import SO3, SO3Tangent

def main():
    # convert yaw to SO3 representation
    yaw = cs.MX.sym('yaw', 1)

    rpy = cs.vertcat(0, 0, np.pi)
    so3 = SO3.from_euler(rpy)
    yaw_tar = 0.4*np.pi
    rpy_tar = cs.vertcat(0, 0, yaw_tar)
    so3_tar = SO3.from_euler(rpy_tar)

    diff_x = so3 - so3_tar
    # quat version
    # d(q1, q2) = 1 - <q1, q2>^2
    diff = 1 - cs.power(cs.mtimes(cs.transpose(so3.as_quat().coeffs()), so3_tar.as_quat().coeffs()), 2)
    print(diff)

if __name__ == '__main__':
    yaw_vec = np.arange(-3*np.pi, 3*np.pi, 0.1)
    print(yaw_vec)