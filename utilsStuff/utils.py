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

def generate_random_positive_definite_matrix(n):
    # 生成一个随机矩阵
    A = np.random.rand(n, n)
    # 通过 A 和 A 的转置相乘来生成正定矩阵
    positive_definite_matrix = np.dot(A, A.T)
    return positive_definite_matrix

def is_positive_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)
