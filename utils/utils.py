import numpy as np


def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])

def normalize_angle(angle):
    while angle > np.pi:
        angle -= np.pi

    while angle <= -np.pi:
        angle += np.pi

    return angle
