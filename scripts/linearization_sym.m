syms w [3 1]
syms v [3 1]
syms m [1 1]
syms I_b [3 3]
syms theta [1 1]
M = m * eye(3);
J_b = [I_b, zeros(3,3);
    zeros(3,3), M];
zeta = [w; v];
v_hat = skew(v);
w_hat = skew(w);

ad_zeta = -[w_hat, v_hat;
            zeros(3,3), w_hat];
% ad_zeta = -[w_hat, zeros(3,3);
%             zeros(3,3), w_hat];

f = ad_zeta * J_b * zeta;

fdx = jacobian(f,zeta)

temp = [skew(I_b * w), m * v_hat;
        m*v_hat, zeros(3,3)];

sol = ad_zeta * J_b + temp

R = [cos(theta), - sin(theta);
       sin(theta), cos(theta)];

RT = inv(R)

function wx = skew(w)
wx = [0, -w(3),  w(2);...
      w(3),  0, -w(1);...
      -w(2), w(1), 0];
end