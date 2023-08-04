import casadi as ca
import numpy as np

# t = ca.MX.sym('t', 2)
# A = ca.vertcat(ca.horzcat(1, 0, 0, 0),
#                ca.horzcat(1, t, t ** 2, t ** 3),
#                ca.horzcat(0, 1, 0, 0),
#                ca.horzcat(0, 1, 2 * t, 3 * t ** 2))
#
# b = ca.vertcat(0, 1, 0, 0)
#
# A_func = ca.Function('A_func', [t], [A])
#
#
# x = ca.solve(A, b)
#
# x_func = ca.Function('x', [t], [x])

def setup_flatness_output():
    t = ca.MX.sym('t')
    a = ca.MX.sym('a', 4)
    b = ca.MX.sym('b', 4)
    z1 = a[0] * t ** 3 + a[1] * t ** 2 + a[2] * t + a[3]
    z2 = b[0] * t ** 3 + b[1] * t ** 2 + b[2] * t + b[3]
    z1_dot = ca.jacobian(z1, t)
    z2_dot = ca.jacobian(z2, t)
    z1_ddot = ca.jacobian(z1_dot, t)
    z2_ddot = ca.jacobian(z2_dot, t)
    z = ca.vertcat(z1, z2)
    z_dot = ca.vertcat(z1_dot, z2_dot)
    z_ddot = ca.vertcat(z1_ddot, z2_ddot)
    flatness_output = {'z': z, 'z_dot': z_dot, 'z_ddot': z_ddot, 'var': {'a': a, 'b': b, 't': t}}
    return flatness_output

# def flat_output_to_state()

t0 = ca.MX.sym('t0')
tf = ca.MX.sym('tf')
t = ca.vertcat(t0, tf)
A = ca.vertcat(ca.horzcat(t0 ** 3, t0 ** 2, t0, 1),     # init pos
               ca.horzcat(tf ** 3, tf ** 2, tf, 1),     # final pos
               ca.horzcat(3 * t0 ** 2, 2 * t0, 1, 0),   # init vel
               ca.horzcat(3 * tf ** 2, 2 * tf, 1, 0))   # final vel
z0 = ca.MX.sym('z0')
z0_dot = ca.MX.sym('z0_dot')
zf = ca.MX.sym('zf')
zf_dot = ca.MX.sym('zf_dot')
b = ca.vertcat(z0, zf, z0_dot, zf_dot)

a = ca.solve(A, b)
coe_func = ca.Function('a_func', [t, b], [a])
time = [0, 1]
b1_value = [0, 1, 0, 0]
b2_value = [0, 1, 0, 00]
a1_coff = coe_func(time, b1_value)
a2_coff = coe_func(time, b2_value)
flatness_output = setup_flatness_output()
z = flatness_output['z']
a = flatness_output['var']['a']
b = flatness_output['var']['b']
t = flatness_output['var']['t']

z_func = ca.Function('z_func', [t, a, b], [z])

t = np.linspace(0, 1)
z_val = np.zeros((2, len(t)))
for i in range(len(t)):
    z_val[0, i] = t[i]
    z_val[0, i] = 0.5 * t[i] ** 2 + 0.5 * t[i]
    z_val[0, i] = 0.8 * t[i] ** 3 + 0.1 * t[i] ** 2 + 0.1 * t[i]
    z_val[1, i] = z_func(t[i], a1_coff, a2_coff)[1]

import matplotlib.pyplot as plt
plt.plot(z_val[0, :], z_val[1, :])
plt.show()



