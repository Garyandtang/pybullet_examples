import numpy as np
import matplotlib.pyplot as plt


edmpc_position_error = np.load("edmpc_position_error.npy")
nmpc_position_error = np.load("nmpc_position_error.npy")
fb_position_error = np.load("fb_position_error.npy")

edmpc_orientation_error = np.load("edmpc_orientation_error.npy")
nmpc_orientation_error = np.load("nmpc_orientation_error.npy")
fb_orientation_error = np.load("fb_orientation_error.npy")

# plot

plt.figure()
plt.plot(edmpc_position_error.T[:500, :], label='edmpc')
plt.title("GMPC position error")
plt.ylim(0, 0.4)
# grid on
plt.grid()
plt.xlabel("$N$")
plt.ylabel("position error")
plt.savefig("edmpc_position_error.png")
plt.show()

plt.figure()
plt.plot(edmpc_orientation_error.T[:500, :], label='edmpc')
plt.title("GMPC orientation error")
plt.ylim(0, 1.5)
plt.grid()
plt.xlabel("$N$")
plt.ylabel("orientation error")
plt.savefig("edmpc_orientation_error.png")
plt.show()

plt.figure()
plt.plot(nmpc_position_error.T[:500, :], label='nmpc')
plt.ylim(0, 0.4)
plt.grid()
plt.title("NMPC position error")
plt.xlabel("$N$")
plt.ylabel("position error")
plt.savefig("nmpc_position_error.png")
plt.show()

plt.figure()
plt.plot(nmpc_orientation_error.T[:500, :], label='nmpc')
plt.title("NMPC orientation error")
plt.ylim(0, 1.5)
plt.grid()
plt.xlabel("$N$")
plt.ylabel("orientation error")
plt.savefig("nmpc_orientation_error.png")
plt.show()

plt.figure()
plt.plot(fb_position_error.T[:500, :], label='fb')
plt.ylim(0, 0.4)
plt.grid()
plt.title("FLC position error")
plt.xlabel("$N$")
plt.ylabel("position error")
plt.savefig("fb_position_error.png")
plt.show()

plt.figure()
plt.plot(fb_orientation_error.T[:500, :], label='fb')
plt.title("FLC orientation error")
plt.ylim(0, 1.5)
plt.grid()
# use latex
plt.xlabel("$N$")
plt.ylabel("orientation error")
plt.savefig("fb_orientation_error.png")
plt.show()

