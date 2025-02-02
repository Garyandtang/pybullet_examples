import numpy as np


def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def smallAdjoint(w, v):
    return np.vstack([np.hstack([skew(w), np.zeros((3, 3))]),
                      np.hstack([skew(v), skew(w)])])


def smallAdjointInv(w, v):
    res = np.zeros((6, 6))
    res[0:3, 0:3] = -skew(w)
    res[3:6, 3:6] = -skew(w)
    res[0:3, 3:6] = -skew(v)
    return res

def adjoint(twist):
    omega = twist[0:3]
    v = twist[3:6]
    omega_hat = skew(omega)
    v_hat = skew(v)
    adj = np.zeros((6, 6))
    adj[0:3, 0:3] = omega_hat
    adj[3:6, 3:6] = omega_hat
    adj[3:6, 0:3] = v_hat
    return adj

def coadjoint(twist):
    omega = twist[0:3]
    v = twist[3:6]
    omega_hat = skew(omega)
    v_hat = skew(v)
    coadj = np.zeros((6, 6))
    coadj[0:3, 0:3] = -omega_hat
    coadj[3:6, 3:6] = -omega_hat
    coadj[0:3, 3:6] = -v_hat
    return coadj

def gamma_right(I, m, omega, v):
    I_omega_hat = skew(I @ omega)
    v_hat = skew(v)
    res = np.zeros((6, 6))
    res[0:3, 0:3] = I_omega_hat
    res[3:6, 0:3] = m * v_hat
    res[0:3, 3:6] = m * v_hat
    return res