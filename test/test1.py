#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:29:50 2023

@author: dr
"""

import numpy as np # Version 1.26.0
import matplotlib.pyplot as plt
import scipy as sp

A = np.array(32) # 0D-array
print("0D",A)
A = np.array((1,2)) # 1D-array
print("1D",A)
print(type(A))

A = np.array( [[1,2],
               [3,4]]) # 2D-array
print("2D",A)
A = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]) # 3D-array
print("3D",A)

print(A.ndim) # Dimension finden


arr1 = np.array([1, 2, 3, 4], ndmin=5) # array mit 5 Dimensionen
arr = arr1
print(arr)

print(arr[0]) # Teil plotten

print(arr[0] + arr[0]) # Addition plotten

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('2nd element on 1st row: ', arr[0, 1]) # plotten 2D array
print('2nd element on 1st row: ', arr1[0, 0,0,0,1]) # 5D-array access
print('2nd element on 1st row: ', arr1[0, 0,0,0,-1]) # 5D-array access (negative index)

# 1 gibt 2. Wert zurück (0 ist erster Wert)
# -1 gibt letzten Wert zurück

print(arr[0][0:2])

arr = np.array([1, 2, 3, 4, 5, 6, 7])
print(arr[4:]) # von 4 (inklusive)
print(arr[:4]) # bis 4 (exklsuive)
print(arr[-3:-1]) # von inklusive drittletztem Element bis letztem Element
print(arr[1:5:2]) # mit step-size (von 1 bis 5, jedes 2. Element)
print(arr[::2]) # alle 2 vom kompletten array 

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[:, 1:4]) # nimm aus äußerer Dimension alles, bei innerer aber nur 2. bis 5. Wert
print(arr[1,0:2 ])

arr = np.array([1, 2, 3, 4])
print(arr.dtype) # Typ herausfinden
arr = np.array(['apple', 'banana', 'cherry'])
print(arr.dtype)
arr = np.array([1, 2, 3, 4], dtype='S')
print(arr)
print(arr.dtype)

arrn = arr.astype('f')
print(arrn) # arrn ist jetzt float

arr = np.array([1, 2, 3, 4, 5])
x = arr.copy() # Kopie, beeinflusst Original nicht
arr[0] = 42
print(x.base) # None = x ist Besitzer
print(arr)
print(x)

arr = np.array([1, 2, 3, 4, 5])
x = arr.view() # View, beeinflusst Original schon
arr[0] = 42
print(x.base) # arr wird geplottet = x ist nicht Besitzer
print(arr)
print(x)



A = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(A.shape) # Aussehen/Dimensionen

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)
print(newarr)
newarr = arr.reshape(2, 2, 3) # der Befehl greift als View darauf zu
print(newarr)

newarr = arr.reshape(4,-1) # bei unbekannten Dimensionen kann -1 eingefügt werden
print(newarr)
# mit nur arr.reshape(-1) kann man die arrays flach machen (Vektor)

for x in A:
   #print(x)
    for y in x:
       #print(y)
        for z in y: # durch jedes Element gehen
            print(z)
            
for x in np.nditer(A):
    print(x) # geht schneller
    
for idx, x in np.ndenumerate(arr):
    print(idx, x) # nummerieren
    
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
for idx, x in np.ndenumerate(arr):
    print(idx, x) # nummerieren 2D
  

# b = np.array([9,9])

# c = A @ b
  



# plt.plot( b , c , label='abc')
# plt.legend() 
# plt.show()

