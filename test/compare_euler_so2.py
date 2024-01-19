import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-20,20,5000)
y = np.sin(x)
z = np.cos(x)
fig = plt.figure()
font_size = 18
line_width = 0.8

ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='blue')
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])
# x, y, z labels
ax.set_xlabel('x',fontsize=font_size)
ax.set_ylabel('$\sin(x)$',fontsize=font_size)
ax.set_zlabel('$\cos(x)$',fontsize=font_size)
ax.grid(linewidth=2)  # Adjust grid line width
plt.tight_layout()
ax.tick_params(labelsize=16)  # Adjust font size of grid numbers
plt.savefig('lie_group.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(-np.pi, np.pi, 10)
y = x
z = 0*x
for i in range(-5,5):
    ax.plot(x+i*2*np.pi, y, z, color='blue')
    ax.scatter(-np.pi+i*2*np.pi, -np.pi, edgecolor='black', facecolors='none')
ax.set_xlabel('x', fontsize=font_size)
ax.set_ylabel('$\\theta$', fontsize=font_size)

ax.set_ylim([-4,4])
ax.set_zlim([-4,4])
plt.tight_layout()
ax.grid(linewidth=3)  # Adjust grid line width
ax.tick_params(labelsize=16)  # Adjust font size of grid numbers

plt.savefig('euler.png')

plt.show()