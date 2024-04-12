import numpy as np
import scipy.linalg as la
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent, SO3Tangent, SO3

def time_varying_lqr(A_list, B_list, Q_list, R_list):
    """
    Time-varying Linear Quadratic Regulator (LQR) algorithm.

    Inputs:
    - A_list: List of state matrices A(t).
    - B_list: List of control matrices B(t).
    - Q_list: List of state cost matrices Q(t).
    - R_list: List of control cost matrices R(t).

    Returns:
    - K_list: List of time-varying control gains K(t).
    """
    num_steps = len(A_list)
    K_list = []

    for t in range(num_steps - 1, -1, -1):
        A = A_list[t]
        B = B_list[t]
        Q = Q_list[t]
        R = R_list[t]

        S_next = np.zeros_like(Q)
        if t < num_steps - 1:
            S_next = K_list[-1].T @ A @ K_list[-1] + Q - (K_list[-1].T @ B + R) @ la.inv(
                K_list[-1].T @ B @ K_list[-1] + R) @ (K_list[-1].T @ B @ K_list[-1]).T

        K = -la.inv(B.T @ S_next @ B + R) @ B.T @ S_next @ A
        K_list.append(K)

    K_list.reverse()
    return K_list


# # Example usage
# A_list = [np.array([[1, 1], [0, 1]]) for _ in range(5)]
# B_list = [np.array([[0], [1]]) for _ in range(5)]
# Q_list = [np.eye(2) for _ in range(5)]
# R_list = [np.eye(1) for _ in range(5)]
#
# K_list = time_varying_lqr(A_list, B_list, Q_list, R_list)
#
# # Storing all K values in a NumPy array
# K_array = np.array(K_list)

a = np.array([0.1, 0.2, 0.3])
b = np.array([0.4, 0.5, 0.6])
A = SO3Tangent(a).hat()
B = SO3Tangent(b).hat()

Ba = B @ a
BA = B @ A

print("Ba: ", SO3Tangent(Ba).hat())
print("BA: ", BA)
