# This python code creates the matrices for a 4 node element

# import packages
import numpy as np # Version 1.26.0
import matplotlib.pyplot as plt
#import scipy as sp
import numpy.linalg as lin
from gauss_points import *
#from element import *
from el import *

# def Ki(x,E,v,th,Nint,Nelem,gapo):
#     # return stiffness matrix for Q4 element
#     # x = coordinates matrix
#     # E = elasticity module
#     # v = Poisson's ratio
#     # th = thickness
#     # Nint = number of Gauss points (integration points)
#     # Nelem = number of points on element
#     # gapo = Gauss-point parameters in form of dictionary {"t": t.flatten(),"s": s.flatten(), "W": W.flatten()}
#     # _______________________
#     # Ei = elasticity matrix for plane strain for element i (mostly the case for rectangular elements)
#     Ei = (E*(1-v))/((1+v)*(1-2*v))*np.array([[1, v/(1-v), 0],
#                                              [v/(1-v), 1, 0],
#                                              [0, 0, (1-2*v)/(2*(1-v))]])
#     # [a] matrix that connects strain and displacement derivatives
#     a = np.array([[1, 0, 0, 0],
#                   [0, 0, 0, 1],
#                   [0, 1, 1, 0]])
#     # get all points
#     t = gapo["t"] # get t
#     s = gapo["s"] # get s
#     # get weights
#     w = gapo["W"] # weights to all corresponding points
#     if Nelem == 4:
#         # calc all parameters for the X and Y functions (Q4)
#         # total Ki for Q4 element
#         A = np.array([[1,-1,-1,1],
#                       [1,1,-1,-1],
#                       [1,1,1,1],
#                       [1,-1,1,-1]])
#         ax = np.linalg.inv(A)@np.transpose(x[0])
#         ay = np.linalg.inv(A)@np.transpose(x[1])
#         # total Ki for Q4 element
#         Ki = np.zeros([8,8])
#         for i in range(0,Nint): # for all Gauss points (N in total)
#             # [J] Jacobi matrix (only Q4)
#             J = np.array([[ax[1]+ax[3]*t[i], ay[1]+ay[3]*t[i]],
#                           [ax[2]+ax[3]*s[i], ay[2]+ay[3]*s[i]]])
#             # make inverse of Jacobi
#             invJ = np.linalg.inv(J)
#             # [b] connects displacement derivatives (Q4)
#             b = np.array([[invJ, np.zeros([2,2])],
#                           [np.zeros([2,2]), invJ]])
#             # make [b] what it should actually look like
#             b = b.transpose(0,2,1,3).reshape(4,4)
#             # [h] as a temporary matrix
#             h = np.array([[-1/4*(1-t[i]), 0, 1/4*(1-t[i]), 0, 1/4*(1+t[i]), 0, -1/4*(1+t[i])],
#                           [-1/4*(1-s[i]), 0, -1/4*(1+s[i]), 0, 1/4*(1+s[i]), 0, 1/4*(1-s[i])]])
#             # assemble [c] differentiated shapefunctions (Q4)
#             c = np.vstack([np.hstack([h, np.zeros([2,1])]), np.hstack([np.zeros([2,1]), h])])
#             # [B] for all different s and t
#             Bi = a @ b @ c
#             Kii = np.transpose(Bi) @ Ei @ Bi * lin.det(J) * th * w[i]
#             Ki = Ki+Kii
#     elif Nelem == 8:
#         print("WIP")
#         raise SystemExit()
#         # calc all parameters for X and Y (Q8)
#     return Ki

# def solve_dis(x,E,v,th,Nint,Nelem,ir,P):
#     # solves for displacement vector
#     # x = coordinates matrix
#     # E = elasticity module
#     # v = Poisson's ratio
#     # th = thickness
#     # Nint = number of Gauss points (integration points)
#     # Nelem = number of points on element
#     # ir = degrees of freedom (0 = free)
#     # P = force vector
#     # _______________________
#     gapo = gauss_points2D(Nint)
#     K = Ki(x,E,v,th,Nint,Nelem,gapo) # get Ki
#     Ndof = 8 # number of degrees of freedom for Q4
#     for i in range(0,Ndof):
#         if ir[i] == 0:
#             for j in range(0,Ndof):
#                 K[i][j] = 0
#                 K[j][i] = 0
#             K[i][i] = 1
#     dis = lin.solve(K,np.transpose(P))
#     return dis

# def solve_P(x,E,v,th,Nint,Nelem,ir,dis):
#     # solves for force vector
#     # x = coordinates matrix
#     # E = elasticity module
#     # v = Poisson's ratio
#     # th = thickness
#     # Nint = number of Gauss points (integration points)
#     # Nelem = number of points on element
#     # ir = degrees of freedom (0 = free)
#     # dis = displacement vector
#     # _______________________
#     gapo = gauss_points2D(Nint)
#     K = Ki(x,E,v,th,Nint,Nelem,gapo) # get Ki
#     Ndof = 8 # number of degrees of freedom for Q4
#     for i in range(0,Ndof):
#         if ir[i] == 0:
#             for j in range(0,Ndof):
#                 K[i][j] = 0
#                 K[j][i] = 0
#             K[i][i] = 1
#     P = K @ dis
#     return P

# try solving for displacements
sigma = np.array([200,100,100]) # kPa
x = np.array([[10.0,0.,0.,10.],
              [5.,5.,0.,0.]]) # cm
E = 200.*1000 # kPa
v = 0.3 # no unit
th = 1 # cm
ir = np.array([1,1,1,1,0,0,1,0]) # DOFs
P = np.array([0,0,0,sigma[1]*0.1*th/2,sigma[0]*0.05*th,sigma[1]*0.05*th,sigma[2]*0.05*th,sigma[2]*0.05*th]) # force vector
Nint = 4 # 4 gauss points
Nelem = 4 # 4 element points
# dis = solve_dis(x,E,v,th,Nint,Nelem,ir,P)
# print(dis)

# # try solving for force
# Pnew = solve_P(x,E,v,th,Nint,Nelem,ir,dis)
# print(P)
# print(Pnew)
# P = K@U
# print(P)

# E = 30000 # MPa
# v = 0.2 # no unit
# x = np.array([[-1,-1.,1,1,-1,-1,1,1],
#               [-1,-1.,-1,-1,1,1,1,1],
#               [-1,1.,1,-1,-1,1,1,-1]]) # cm
# hexa = []
# for i in range(len(x[0])):
#     hexa.append(nodes([x[0][i],x[1][i],x[2][i]],i))
# K = np.zeros((24,24))
# P = np.zeros([24,1])
# U = P
# elem = Element("H8",1) # create element
# elem.setNodes(hexa) # set coordinates
# elem.setMaterial(["E","v"],np.array([E,v])) # set Material
# elem.computeYourself(K,P,U,0,0,0)
# print(np.max(K-K.T))

# hexa = []
# for i in range(len(x[0])):
#     hexa.append(nodes([x[0][i],x[1][i]],i))
# K = np.zeros((8,8))
# P = np.zeros((8))
# U = np.zeros((8,1)) # for t = 0
# dU = np.array([[1],[2],[1],[2],[0],[0],[0],[0]])
# elem = Element("Quad4E",1) # create element
# elem.setNodes(hexa) # set coordinates
# elem.setProperties([1])
# elem.setMaterial("linearelastic",np.array([E,v])) # set Material
# time = 0
# dTime = np.array([0,0,20,20,20,20,20])
# for t in range(len(dTime)):
#     time = np.array([dTime[t], np.sum(dTime[0:t])])
#     U += dU * dTime[t]
#     elem.computeYourself(K,P,U,dU,time,dTime)

# h = [3, 4, 5]
# x = h
# x = [y+1 for y in x]
# print(x)
# print(h)

# l = [1, 5, 9, 3]
# h = l # gleiche Identität

# h[0], h[2] = h[2], h[0] # Identität unverändert
# h[1] += 5 # Identität unverändert
# h[3] = 6 # Identität unverändert

# print(h)  # [9, 5, 1, 3]
# print(l)  # [9, 5, 1, 3]

# print(id(h), id(l))
# h *= 2 # Identität bleibt erhalten
# print(id(h), id(l))
# h = h * 2 # Identität wird gelöscht, h neu
# print(id(h), id(l))

# print(h)  # [9, 5, 1, 3, 9, 5, 1, 3, 9, 5, 1, 3]
# print(l)  # [9, 5, 1, 3, 9, 5, 1, 3]

# dic = gauss_points3D(27)
# z = dic["z"]/np.sqrt(0.6)
# t = dic["t"]/np.sqrt(0.6)
# s = dic["s"]/np.sqrt(0.6)
# w = dic["W"]
# print(z)
# print(t)
# print(s)
# print(w)

# z = np.sqrt(0.6) * np.array([-1, 0, 1]) # get part of z
# s = np.sqrt(0.6) * np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])  # get part of s
# w1 = (5 / 9) ** 3
# w2 = (5 / 9) ** 2 * (8/9)
# w3 = (5/9) * (8/9) ** 2
# w4 = (8/9) ** 3
# w = np.array([w1, w2, w1, w2, w3, w2, w1, w2, w1, w2, w3, w2, w3])
# diction = {
#         "plStrain": False,
#         "Nint": 27,
#         "z": np.hstack([z, z, z, z, z, z, z, z, z]),
#         "t": np.sqrt(0.6) * np.hstack([-np.ones(9), np.zeros(9), np.ones(9)]),
#         "s": np.hstack([s, s, s]),
#         "w": np.hstack([w, w4, w[::-1]]),
#     }
dim = 3
b = np.zeros([2, dim * dim, dim * dim])
invJ = np.array([[1,1,1],
                 [1,1,1],
                 [1,1,1]])
b[:][:][0] = np.bmat([[invJ, np.zeros([3, 3]), np.zeros([3, 3])],
                      [np.zeros([3, 3]), invJ, np.zeros([3, 3])],
                      [np.zeros([3, 3]), np.zeros([3, 3]), invJ]])
print(b)