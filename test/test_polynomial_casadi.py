import casadi as ca
from matplotlib import pyplot as plt
import numpy as np
# Define symbolic variables
t = ca.SX.sym('t')
a = ca.SX.sym('a')
b = ca.SX.sym('b')
c = ca.SX.sym('c')
d = ca.SX.sym('d')

# Define the polynomial function
y = a*t**3 + b*t**2 + c*t + d

# Define a function to compute the gradient of the polynomial function
dydt = ca.jacobian(y, t)

# Create a function that can be evaluated and differentiated
f = ca.Function('f', [t, a, b, c, d], [y, dydt])

# Evaluate the function and its gradient at time t=1 with coefficients a=1, b=2, c=3, d=4
t_val = 1.0
a_val = 1.0
b_val = 1.0
c_val = 1.0
d_val = 1.0
y_val, dydt_val = f(t_val, a_val, b_val, c_val, d_val)

# Print the function value and its gradient
print(f"Function value at t={t_val}: {y_val}")
print(f"Gradient value at t={t_val}: {dydt_val}")


# plot the function
t_val = np.linspace(-1, 1, 100)
y_val = np.zeros_like(t_val)
dydt_val = np.zeros_like(t_val)
for i in range(len(t_val)):
    y_val[i], dydt_val[i] = f(t_val[i], a_val, b_val, c_val, d_val)

plt.figure()
plt.plot(t_val, y_val, label='y')
plt.plot(t_val, dydt_val, label='dydt')
plt.legend()
plt.show()