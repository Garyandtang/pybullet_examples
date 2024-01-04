import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axis
fig, ax = plt.subplots()

# Iterate through the data points and plot step-by-step
for i in range(len(x)):
    ax.plot(x[:i+1], y[:i+1], 'b-')
    plt.pause(0.1)  # Pause for a short duration to visualize step-by-step
    plt.show()

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Step-by-Step Plot')

# Display the final plot
plt.show()