# this python code creates the matrices for a 4 node element

# import packages
import numpy as np # Version 1.26.0
import matplotlib.pyplot as plt
import scipy as sp
import numpy.linalg as lin

def Ki(x,E,v,th,w_s,w_t):
    # x = coordinates matrix
    # E = elasticity module
    # v = Poisson's ratio
    # t = thickness
    # w_s = weight factor for quadrature for parameter s
    # w_t = weight factor for quadrature for parameter t
    # Ei = elasticity matrix for plane strain for element i (mostly the case for Q4 elements)
    Ei = (E*(1-v))/((1+v)*(1-2*v))*np.array([[1, v/(1-v), 0],
                                             [v/(1-v), 1, 0],
                                             [0, 0, (1-2*v)/(2*(1-v))]])
    # [a] matrix that connects strain and displacement derivatives
    a = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 1, 1, 0]])
    s = 1/np.sqrt(3)*np.array([-1, 1, 1,-1])
    t = 1/np.sqrt(3)*np.array([-1,-1, 1, 1])
    # calc all parameters for the X and Y functions
    a2 = -x[0][0]+x[0][1]+x[0][2]-x[0][3]
    a3 = -x[0][0]-x[0][1]+x[0][2]+x[0][3]
    a4 = +x[0][0]-x[0][1]+x[0][2]-x[0][3]
    a6 = -x[1][0]+x[1][1]+x[1][2]-x[1][3]
    a7 = -x[1][0]-x[1][1]+x[1][2]+x[1][3]
    a8 = +x[1][0]-x[1][1]+x[1][2]-x[1][3]
    # total Ki for Q4 element
    Ki = np.zeros([8,8])
    for i in range(0,3):
        # [J] Jacobi matrix
        J = np.array([[a2+a4*t[i], a6+a8*t[i]],
                      [a3+a4*s[i], a7+a8*s[i]]])
        # [b] connects displacement derivatives
        b = np.array([[np.linalg.inv(J), np.zeros([2,2])],
                      [np.empty([2,2]), np.linalg.inv(J)]])
        # make [b] what it should actually look like
        b = b.transpose(0,2,1,3).reshape(4,4)
        # [c] differentiated shapefunctions
        c = np.array([[-1/4*(1-t[i]), 0, 1/4*(1-t[i]), 0, 1/4*(1+t[i]), 0, -1/4*(1+t[i]), 0],
                      [-1/4*(1-s[i]), 0, -1/4*(1+s[i]), 0, 1/4*(1+s[i]), 0, -1/4*(1-s[i]), 0],
                      [0, -1/4*(1-t[i]), 0, 1/4*(1-t[i]), 0, 1/4*(1+t[i]), 0, -1/4*(1+t[i])],
                      [0, -1/4*(1-s[i]), 0, -1/4*(1+s[i]), 0, 1/4*(1+s[i]), 0, -1/4*(1-s[i])]])
        # [B] for all different s and t
        Bi = a @ b @ c
        Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * th * w_s * w_t
        Ki = Ki+Kii
    return Ki

def solve_dis(x,E,v,th,w_s,w_t,ir,P):
    # solves for displacement vector
    # x = coordinates matrix
    # E = elasticity module
    # v = Poisson's ratio
    # t = thickness
    # w_s = weight factor for quadrature for parameter s
    # w_t = weight factor for quadrature for parameter t
    # ir = degrees of freedom (0 = free)
    K = Ki(x,E,v,th,w_s,w_t) # get Ki
    Ndof = 8
    for i in range(0,Ndof):
        if ir[i] == 0:
            for j in range(0,Ndof):
                K[i][j] = 0
                K[j][i] = 0
            K[i][i] = 1
    dis = lin.solve(K,np.transpose(P))
    return dis

def solve_P(x,E,v,th,w_s,w_t,ir,dis):
    # solves for force vector
    # x = coordinates matrix
    # E = elasticity module
    # v = Poisson's ratio
    # t = thickness
    # w_s = weight factor for quadrature for parameter s
    # w_t = weight factor for quadrature for parameter t
    # ir = degrees of freedom (0 = no displacement)
    K = Ki(x,E,v,th,w_s,w_t) # get Ki
    Ndof = 8
    for i in range(0,Ndof):
        if ir[i] == 0:
            for j in range(0,Ndof):
                K[i][j] = 0
                K[j][i] = 0
            K[i][i] = 1
    P = K @ dis
    return P

# try solving for displacements
sigma = np.array([200,100,100]) # kPa
x = np.array([[10.0,0.,0.,10.],
              [5.,5.,0.,0.]]) # cm
E = 200.*1000 # kPa
v = 0.3 # no unit
th = 1 # cm
w_s = 1 # Gauss parameter 1
w_t = 1 # Gauss parameter 1
ir = np.array([1,1,1,1,0,0,1,0]) # DOFs
P = np.array([0,0,0,sigma[1]*0.1*th/2,sigma[0]*0.05*th,sigma[1]*0.05*th,sigma[2]*0.05*th,sigma[2]*0.05*th]) # force vector
dis = solve_dis(x,E,v,th,w_s,w_t,ir,P)
print(dis)

# try solving for force
Pnew = solve_P(x,E,v,th,w_s,w_t,ir,dis)
print(P)
print(Pnew)
