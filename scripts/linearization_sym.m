syms w [3 1]
syms v [3 1]
syms m [1 1]
syms I_b [3 3]
M = m * eye(3);
J_b = [I_b, zeros(3,3);
    zeros(3,3), M];
zeta = [w; v];
v_hat = [0 -v(3) v(2);
         v(3) 0 -v(1);
         -v(2) v(1) 0];
w_hat = [0 -w(3) w(2);
         w(3) 0 -w(1);
         -w(2) w(1) 0];

ad_zeta = -[w_hat, v_hat;
            zeros(3,3), w_hat];

f = ad_zeta * J_b * zeta;

fdx = jacobian(f,zeta)

temp = [skew(I_b * w), m * v_hat;
        m*v_hat, zeros(3,3)];

sol = ad_zeta * J_b + temp

ccode(fdx)


function wx = skew(w)
wx = [0, -w(3),  w(2);...
      w(3),  0, -w(1);...
      -w(2), w(1), 0];
end