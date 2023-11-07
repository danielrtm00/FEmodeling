#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:30:48 2023

@author: dr
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2)) # kombinieren zweier Vektoren
print(arr)

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1) # 2D, man muss Achse übergeben
print(arr)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=1) # mit stack Funktion
print(arr)

# hstack für stack along rows und vstack along columns und dstack along height

arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3) # gibt 3 arrays zurück
print(newarr)
print(newarr[0]) # aufgeteilte arrays erreichen
print(newarr[1])
print(newarr[2])

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3) # 2D arrays aufgeteilt
print(newarr)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1) # andere Aufteilung
print(newarr)
# auch hier Lösungen mit hsplit, vsplit und dsplit möglich


arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4) # finde Elemente mit =4 (index)
print(x)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x = np.where(arr%2 == 0) # gerade zahlen finden (geteilt durch 2 mit Rest 0)
x = np.where(arr%2 == 1) # gerade zahlen finden (geteilt durch 2 mit Rest 1)

arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7) # gibt den Index zurück an dem arr = 7
print(x)

arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7, side='right') # zählt den Index rückwärts
print(x)

arr = np.array([1, 3, 5, 7])
x = np.searchsorted(arr, [2, 4, 6]) # gibt Indizes zurück wo 2, 4, 6 einsortiert werden sollten
print(x)

arr = np.array([3, 2, 0, 1])
print(np.sort(arr)) # sortiert numerisch

arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr)) # sortiert alphabetisch

arr = np.array([True, False, True])
print(np.sort(arr)) # sortiert boolean (False vorne)

arr = np.array([41, 42, 43, 44])
x = [True, False, True, False]
newarr = arr[x] # findet true Ergebnisse
print(newarr)
newarr = arr[np.invert(x)] # findet false Ergebnisse
print(newarr)

filter_arr = []
# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

filter_arr = arr > 42 # schnellere Methode
print(filter_arr)

filter_arr = arr%2 == 0 # schnellere Methode für durch 2 dividieren
print(filter_arr)

##### RANDOM (nur grundlegendes)
from numpy import random
import seaborn as sns
x = random.randint(100) # von 0 bis 100, integer
x = random.rand() # von 0 bis 1
x = random.randint(100, size=(20)) # 20 Werte in array-Form
x = random.rand(20) # 20 floats
x = random.choice([3, 5, 7, 9], size=(3, 5)) # Werte aus Auswahl generieren
print(x)

sns.distplot(random.normal(size=1000), hist=False)
plt.show()