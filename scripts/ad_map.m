function res = ad_map(twist)
w = twist(1:3);
v = twist(4:6);
w_hat = skew(w);
v_hat = skew(v);
res = -[w_hat, v_hat;
        zeros(3,3), w_hat];
end