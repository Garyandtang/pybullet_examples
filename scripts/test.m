R = rand(3,3);

wb = rand(3,1)
I = diag(rand(3,1))

ws = R * wb;


a = I * skew(ws) * wb + skew(wb) * R * I *wb


b = skew(wb) * I *wb