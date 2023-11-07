# this python code creates the matrices for a 4 node element

# import packages
import numpy as np # Version 1.26.0
import matplotlib.pyplot as plt
import scipy as sp

def matrices(x,E,v,t):
    # x = coordinates matrix
    # E = elasticity module
    # v = Poisson's ratio
    # t = thickness
    Ei = (E*(1-v))/((1+v)*(1-2*v))*np.array([[1, v/(1-v), 0],
                                             [v/(1-v), 1, 0],
                                             [0, 0, (1-2*v)/(2*(1-v))]]) # elasticity matrix for plane strain for element i
    A = np.array([[1,-1,-1,1],
                  [1,1,-1,-1],
                  [1,1,1,1],
                  [1,-1,1,-1]]) # connects functions
    

def Ki(x,Ei)
    # [a] matrix that connects strain and displacement derivatives
    a = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 1, 1, 0]])
    s = 1/np.sqrt(3)*np.array([-1, 1, 1,-1])
    t = 1/np.sqrt(3)*np.array([-1,-1, 1, 1])
    a1 = np.sum(x[0])
    a2 = -x[0][0]+x[0][1]+x[0][2]-x[0][3]
    a3 = -x[0][0]-x[0][1]+x[0][2]+x[0][3]
    a4 = +x[0][0]-x[0][1]+x[0][2]-x[0][3]
    a5 = np.sum(x[1])
    a6 = -x[1][0]+x[1][1]+x[1][2]-x[1][3]
    a7 = -x[1][0]-x[1][1]+x[1][2]+x[1][3]
    a8 = +x[1][0]-x[1][1]+x[1][2]-x[1][3]
    for i in range(0,3):
        J = np.array([[a2+a4*t[i], a6+a8*t[i]],
                      [a3+a4*s[i], a7+a8*s[i]]])
        # [b] connects displacement derivatives
        b = np.array([[np.linalg.inv(J), np.empty(2,2)],
                      [np.empty(2,2), np.linalg.inv(J)]]
        # [c] differentiated shapefunctions
        c = np.array([[-1/4*(1-t[i]), 0, 1/4*(1-t[i]), 0, 1/4*(1+t[i]), 0, -1/4*(1+t[i]), 0],
                      [-1/4*(1-s[i]), 0, -1/4*(1+s[i]), 0, 1/4*(1+s[i]), 0, -1/4*(1-s[i]), 0],
                      [0, -1/4*(1-t[i]), 0, 1/4*(1-t[i]), 0, 1/4*(1+t[i]), 0, -1/4*(1+t[i])],
                      [0, -1/4*(1-s[i]), 0, -1/4*(1+s[i]), 0, 1/4*(1+s[i]), 0, -1/4*(1-s[i])]])
        Bi = np.dot(np.dot(a,b),c)
        



# initalize sizes of matrices
#x = np.array([[0,1,1,0],
#              [0,0,1,1]]) # testing coordinates
#A = (np.amax(x[0])-np.amin(x[0]))*(np.amax(x[1])-np.amin(x[1])) # calculate area
#
# test values for ny and elasticity module:
#E = 210000.0 # [MPa=N/mm²]
#v = 0.3 # []
#
#Ei = E/(1-v**2)*np.array([[1, v, 0],
#                          [v, 1, 0],
#                          [0, 0, 0.5*(1-v)]]) # for plane stress
#
#Pi = np.empty((1,8)) # empty element force matrix
#
#D = XY(1,1,x[0])
#print(D)
#A = [[1, x[0][0], x[1][0], x[0][0]*x[1][0], 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, x[0][0], x[1][0], x[0][0]*x[1][0]],
#     [1, x[0][1], x[1][1], x[0][1]*x[1][1], 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, x[0][1], x[1][1], x[0][1]*x[1][1]],
#     [1, x[0][2], x[1][2], x[0][2]*x[1][2], 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, x[0][2], x[1][2], x[0][2]*x[1][2]],
#     [1, x[0][3], x[1][3], x[0][3]*x[1][3], 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, x[0][3], x[1][3], x[0][3]*x[1][3]]] # set center


