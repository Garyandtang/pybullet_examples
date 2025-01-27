import numpy as np
from scipy.linalg import cholesky

def make_positive_definite(matrix, epsilon=1e-6):
    try:
        # 尝试Cholesky分解
        A = cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        # 如果失败，添加一个小的正定矩阵
        n = matrix.shape[0]
        identity_matrix = np.eye(n)
        return matrix + epsilon * identity_matrix

# 示例
A = np.array([[1, 1], [2, 1]])
A_positive_definite = make_positive_definite(A)
print(A_positive_definite)