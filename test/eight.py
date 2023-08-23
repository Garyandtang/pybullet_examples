#!/usr/bin/env python

from math import cos, pi, sin, sqrt
# Teleports turtlebot to given location
import numpy as np
import scipy.linalg
import time
import matplotlib.pyplot as plt
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
import casadi as ca
import math
from utils.enum_class import TrajType


# Publishes linear/angular velocity commands to turtlebot
def control_turtle():
    t = 0.0                 # time variable
    T =50
    i = 0
    nTraj = 5000
    store_twist = np.zeros((3, nTraj))
    store_se2 = np.zeros((4, nTraj))
    init_state = np.array([1, 1, 0])
    store_se2[:, 0] = SE2(init_state[0], init_state[1], init_state[2]).coeffs()
    while i < nTraj:
        # calculate linear velocity
        xdot = 1.0*cos(4.0*pi*t/T)*4.0*pi/T
        ydot = 1.0*cos(2.0*pi*t/T)*2.0*pi/T
        v = sqrt(xdot**2+ydot**2)

        # calculate angular velocity
        xdotdot = -1.0*sin(4*pi*t/T)*(4.0*pi/T)**2
        ydotdot = -1.0*sin(2*pi*t/T)*(2.0*pi/T)**2
        w = (ydotdot*xdot - xdotdot*ydot) / (xdot**2 + ydot**2)

        twist = np.array([v, 0, w])
        store_twist[:, i] = twist

        # increment time
        if t == T:
            t = 0.0
        else:
            t = t + 0.01
        i = i + 1

    print("max_linear_velocity: ", np.max(store_twist[0, :]))
    print("max_angular_velocity: ", np.max(store_twist[2, :]))
    print("min_linear_velocity: ", np.min(store_twist[0, :]))
    print("min_angular_velocity: ", np.min(store_twist[2, :]))
    for i in range(nTraj-1):
        twist = store_twist[:, i]
        X = SE2(store_se2[:, i])
        X = X + SE2Tangent(twist * 0.01)
        store_se2[:, i+1] = X.coeffs()

    # plot trajectory
    plt.figure(1)
    plt.plot(store_se2[0, :], store_se2[1, :], 'b')
    plt.plot(store_se2[0, 0], store_se2[1, 0], 'bo')
    plt.plot(store_se2[0, -1], store_se2[1, -1], 'bx')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    control_turtle()