import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits import mplot3d

from liecasadi import SO3, SO3Tangent, SE3Tangent, SE3

opti = cs.Opti()

T = 2
N = 100


pos = opti.variable(3, N + 1)
quat = opti.variable(4, N + 1)
twist = opti.variable(6, N+1)


dt = T / N


for k in range(N):
    curr_SE3 = SE3(pos[:, k], quat[:, k])
    curr_se3 = SE3Tangent(twist[:, k] * dt)
    forward_SE3 = curr_SE3 * curr_se3.exp()
    next_SE3 = SE3(pos[:, k + 1], quat[:, k + 1])
    opti.subject_to(next_SE3.pos == forward_SE3.pos)
    opti.subject_to(next_SE3.xyzw == forward_SE3.xyzw)

cost = 0
for k in range(N):
    cost += cs.sumsqr(twist[:, k])


# set initial pose
opti.subject_to(quat[:, 0] == SO3.Identity().as_quat())
opti.subject_to(pos[:, 0] == np.array([0, 0, 0]).reshape(3, 1))

# set final pose
opti.subject_to(quat[:, N] == SO3.from_euler(np.array([0, 0, 0.9*np.pi])).as_quat())
opti.subject_to(pos[:, N] == np.array([1, 1, 0]).reshape(3, 1))

# # control bounds
# opti.subject_to(opti.bounded(-1, T, 1))


opti.minimize(cost)
opti.solver("ipopt")
try:
    sol = opti.solve()
except:
    print("Can't solve the problem!")

pos_sol = sol.value(pos)

quat_sol = sol.value(quat)
# quat to euler
euler_sol = np.zeros((3, N+1))
for i in range(N+1):
    euler_sol[:, i] = SO3.from_quat(quat_sol[:, i]).as_euler().full().reshape(3, )

print(euler_sol)
# plot euler angle
fig = plt.figure()
plt.plot(euler_sol[2, :], 'y')
plt.title('euler angle')
plt.show()


print(max(pos_sol[2, :]) - min(pos_sol[2, :]))
# plot 3d xyz position
fig = plt.figure()
ax = plt.figure().add_subplot(projection='3d')
# set x, y, z limit
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
ax.set_zlim(-1, 2)
ax.plot(pos_sol[0, :], pos_sol[1, :], pos_sol[2, :])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# plot 2d xy position
fig = plt.figure()
# x limit and y limit
plt.xlim(-1, 2)
plt.ylim(-1, 2)
plt.plot(pos_sol[0, :], pos_sol[1, :])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
