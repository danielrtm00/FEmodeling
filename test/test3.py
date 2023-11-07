#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 09:19:38 2023

@author: dr
"""

#### ufuncs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from math import log
x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = []

for i, j in zip(x, y):
  z.append(i + j) # zusammenzählen
z = np.add(x,y)
print(z)
print(np.transpose(z)) # transponiert

def myadd(x, y):
  return x+y
myadd = np.frompyfunc(myadd, 2, 1)
print(myadd([1, 2, 3, 4], [5, 6, 7, 8])) # add vectors

def myadd1(x, y):
  return np.add(x,y) # addieren
print(myadd1(x,y)) # add in a different way

arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])
# jeweils einzelne Operatoren
newarr = np.subtract(arr1, arr2) # subtract
print(newarr)
newarr = np.multiply(arr1, arr2) # multiply
print(newarr)
newarr = np.divide(arr1,arr2) # divide
print(newarr)
newarr = np.power(arr1,arr2) # power
print(newarr)
newarr = np.mod(arr1,arr2) # remainder
print(newarr)
newarr = np.remainder(arr1,arr2) # remainder
print(newarr)
newarr = np.divmod(arr1,arr2) # remainder and quotient
print(newarr)
newarr = np.abs(arr1) # absolute
print(newarr)

arr = np.trunc([-3.1666, 3.6667]) # round to int closer to 0 but stay with float
print(arr)
arr = np.around(3.1666, 2) # runden auf 2 Stellen
print(arr)
arr = np.floor([-3.1666, 3.6667]) # abrunden auf nächsten niedrigeren int
print(arr)
arr = np.ceil([-3.1666, 3.6667]) # aufrunden auf höheren int
print(arr)

arr = np.arange(1, 10)
print(np.log2(arr)) # base 2 log
print(np.log10(arr)) # base 10 log
print(np.log(arr)) # natural base
nplog = np.frompyfunc(log, 2, 1)
print(nplog(arr, 15)) # base 

newarr = np.sum([arr1,arr2]) # summe aller elemente, mit axis = 0,1 andere Achse
print(newarr)
newarr = np.sum([arr1,arr2],axis=1) # summe aller elemente, mit axis = 0,1 andere Achse
print(newarr)
newarr = np.cumsum([arr1,arr2]) # partielles zusammenzählen in Vektor
print(newarr)

arr = np.array([10, 15, 25, 5])
newarr = np.diff(arr) # Subtraktion mit Nachbarwerten
print(newarr)