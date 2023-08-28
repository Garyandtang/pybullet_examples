import numpy as np
import matplotlib.pyplot as plt


data = np.load("edmpc_position_error.npy")

plt.figure()
plt.plot(data.T, label='edmpc')
plt.title("edmpc position error")
plt.xlabel("N")
plt.ylabel("position error")
plt.show()
print('end')