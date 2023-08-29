import numpy as np
import matplotlib.pyplot as plt

orientation_error = np.load("orientation_error.npy")
position_error = np.load("position_error.npy")

# plot
plt.figure()
plt.plot(position_error, label='position error')
plt.title("position error")
plt.ylim(0, 0.4)
# grid on
plt.grid()
plt.xlabel("$N$")
plt.ylabel("position error")
plt.savefig("position_error.png")
plt.show()

plt.figure()
plt.plot(orientation_error, label='orientation error')
plt.title("orientation error")
plt.ylim(0, 1.5)
plt.grid()
plt.xlabel("$N$")
plt.ylabel("orientation error")
plt.savefig("orientation_error.png")
plt.show()

