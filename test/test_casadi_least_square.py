import casadi as ca

# the parameters to optimise
a = ca.MX.sym('a', 1)
b = ca.MX.sym('b', 1)

# I have F = a*r + b*r**3. I specify the value 'r' symbolically:
r = ca.MX.sym('r', 1)

# the measurements (only one now)
r_m = ca.MX.sym('r_m', 1)
F_m = ca.MX.sym('F_m', 1)

# given a measurement, minimal distance in input and output
Fhat = a*r + b*r**3
dr = (r - r_m)**2
dF = (Fhat - F_m)**2

# find the r (and F) to minimise distance, for given a, b, r_m, F_m
opt = {'x': r, 'f': dr + dF, 'p': ca.vertcat(a, b, r_m, F_m)}
solver = ca.nlpsol('min_dist', 'ipopt', opt)

# make an expression that only returns the distance
min_dist = solver(x0=[0.25], p=ca.vertcat(a, b, r_m, F_m))['f']

# optimise a, b
opt = {'x': ca.vertcat(a, b), 'f': min_dist, 'p': ca.vertcat(r_m, F_m) }
solver = ca.nlpsol('min_param', 'ipopt', opt)

print(2)