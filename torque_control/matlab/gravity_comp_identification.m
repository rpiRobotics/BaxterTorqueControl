clear; clc;
% AUTHOR: Andrew Cunningham
% DESCRIPTION: This script creates a gravity compensation matrix
%              for the Baxter robot given calibration data

ex = [1;0;0]; ey = [0;1;0]; ez = [0;0;1]; zeroV=[0;0;0];
p01 = zeroV; p12=[.069;0;.27035]; p23=zeroV; p34=[.36435;0;-.069];
p45=[.37429;0;-.01]; p56 = zeroV; p67 = zeroV;
P = [p01 p12 p23 p34 p45 p56 p67];

%% DEFINE BAXTER SPECIFICS
h = [ez ey ex ey ex ey ex];

%% DEFINE SYMBOLICS
q = sym('q', [1 7]);
offsets = sym('P',[3 7]);
syms spring1 spring2 spring3 spring4 
springs = sym('K',[2 5]);

g = 9.81;

%% CREATE POTENTIAL ENERGY
Potential = 0;
P_0I = [0; 0; 0];
R = eye(3);
% GRAVITY
for i=1:1:7
    R = R*rot(h(:,i), q(i));
    P_0I = P_0I + R*P(:,i);                % Vector to I'th joint
    P_0Ci = P_0I + R*offsets(:,i);         % Vector to I'th center of mass 
    Potential = Potential+g*ez'*P_0Ci;
end

%% DIFFERENITIATE
G = gradient(Potential, q);
G(2) = G(2) + spring1*q(2)-spring2;
G(4) = G(4) + spring3*q(4)-spring4;

%% FORM SYSTEM OF EQUATIONS
unknowns = [];
for i=1:1:7
    unknowns = [unknowns; offsets(:,i)];
end

unknowns = [unknowns; spring1; spring2; spring3; spring4];

A_template = jacobian(G, unknowns);

%% FIT DATA
data = csvread('pid_auto_big.csv', 1);

A = [];
b = [];
for i=1:1:size(data,1)
    q_data = data(i,1:7);
    t_data = data(i,8:end);
    A = [A; double(subs(A_template, q, q_data))];
    b = [b; t_data'];
end

%% ADD WEIGHTING
%v = [0 1 2 4 2 8 0];
v = [1 1 1 1 1 1 1];
v_diag = [];
for i=1:1:size(A,1)/7
     v_diag = [v_diag v];
end

W = diag(v_diag, 0);
A_w = W*A; b_w = W*b;
%params2 = pinv(A_w)*b_w;

%% COMPUTE FIT WITH SVD
[U,S,V] = svd(A_w);
s = diag(S);
n = size(A_w,2);

r = rank(A_w);
d = U'*b_w;
params = V* ( [d(1:r)./s(1:r); zeros(n-r,1) ] );

%% EVALUATE QUALITY OF FIT
error = 0;
data_verify = csvread('pid_auto_big.csv', 1);
G_m = A_template * params;
diff_history = [];
for i=1:1:size(data_verify,1)
    q_data = data_verify(i,1:7);
    t_data = data_verify(i,8:end);
    diff = t_data' - double(subs(G_m, q, q_data));
    diff_history = [diff_history diff];
    error = error + double(diff'*diff);
end

mse = error/size(data_verify,1)
save('grav_comp', 'G_m', 'q')