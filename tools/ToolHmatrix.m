% tool for getting the h matrix
syms s t z
N = [1/8*(1-s)*(1-t)*(1-z)*(-s-t-z-2);
    0;
    0;
    1/8*(1-s)*(1-t)*(1+z)*(-s-t+z-2);
    0;
    0;
    1/8*(1+s)*(1-t)*(1+z)*(s-t+z-2);
    0;
    0;
    1/8*(1+s)*(1-t)*(1-z)*(s-t-z-2);
    0;
    0;
    1/8*(1-s)*(1+t)*(1-z)*(-s+t-z-2);
    0;
    0;
    1/8*(1-s)*(1+t)*(1+z)*(-s+t+z-2);
    0;
    0;
    1/8*(1+s)*(1+t)*(1+z)*(s+t+z-2);
    0;
    0;
    1/8*(1+s)*(1+t)*(1-z)*(s+t-z-2);
    0;
    0;
    1/4*(1-s)*(1-t)*(1-z^2);
    0;
    0;
    1/4*(1-s^2)*(1-t)*(1+z);
    0;
    0;
    1/4*(1+s)*(1-t)*(1-z^2);
    0;
    0;
    1/4*(1-s^2)*(1-t)*(1-z);
    0;
    0;
    1/4*(1-s)*(1+t)*(1-z^2);
    0;
    0;
    1/4*(1-s^2)*(1+t)*(1+z);
    0;
    0;
    1/4*(1+s)*(1+t)*(1-z^2);
    0;
    0;
    1/4*(1-s^2)*(1+t)*(1-z);
    0;
    0;
    1/4*(1-s)*(1-t^2)*(1-z);
    0;
    0;
    1/4*(1-s)*(1-t^2)*(1+z);
    0;
    0;
    1/4*(1+s)*(1-t^2)*(1+z);
    0;
    0;
    1/4*(1+s)*(1-t^2)*(1-z)];
h(1,:) = simplify(diff(N, s));
h(2,:) = simplify(diff(N, t));
h(3,:) = simplify(diff(N, z));
r = string(h);
k = "h = np.array([[" + newline;
for i = 1:3
    for j = 1:length(r)
        k = k + r(i,j) + "," + newline;
    end
    if i ~= 3
        k = k + "],[" + newline;
    else
        k = k + "]])";
    end
end
writematrix(k, "hMatrix.txt")

