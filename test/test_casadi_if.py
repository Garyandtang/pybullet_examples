import casadi as ca

# Define an input variable for the function
x = ca.MX.sym('x')

# Define a CasADi function that returns the absolute value of x
y = x
abs_func = ca.Function('abs_func', [x], [y])

# Create an optimization problem
opti = ca.Opti()

# Define a constraint using the CasADi function
# c = opti.bounded(0, abs_func(x) - 1, 1)

# Define an objective function using the CasADi function
x = opti.variable(1, 1)
obj = abs_func(x)
opti.minimize(obj)
opts_setting = {'ipopt.max_iter': 80, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-3,
                        'ipopt.acceptable_obj_change_tol': 1e-3}
opti.solver('ipopt', opts_setting)

# Set initial guess for optimization variable
opti.set_initial(x, -1.0)

# Solve the optimization problem
sol = opti.solve()

# Print the solution
print("x = ", sol.value(x))