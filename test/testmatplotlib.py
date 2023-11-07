#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:03:16 2023

@author: dr
"""

import matplotlib.pyplot as plt
import numpy as np
# xpoints = np.array([0, 6])
# ypoints = np.array([0, 250])
# plt.plot(xpoints, ypoints)
# plt.plot(xpoints, ypoints, "o") # as points
# #ypoints = np.array([3, 8, 1, 10, 5, 7])
# plt.plot(ypoints) # with default x-points
# plt.plot(xpoints, ypoints, marker="*") # points and line
# plt.show()
# plt.figure(1)

# plt.figure(2) # 2 gleichzeitig öffnen
# ypoints = np.array([3, 8, 1, 10])
# plt.plot(ypoints, 'o:r') # Punktlinie rot mit o-Marker
# plt.show()

# plt.figure(3)
# plt.plot(ypoints, marker="o", ms = 20, mec = "r") # markersize 20 mit o Marker und roter Umrandung
# plt.show()

# plt.figure(3)
# plt.plot(ypoints, marker="o", ms = 20, mfc = "r") # markersize 20 mit o Marker und roter Kreise mit Standardumrandungsfarbe
# plt.show()

# plt.figure(4)
# plt.plot(ypoints, marker = 'o', ms = 20, mec = '#4CAF50', mfc = '#4CAF50') # grüne Punkte
# plt.show()

# plt.figure(5)
# plt.plot(ypoints, ls = "--") # dashed line
# plt.show()

# plt.figure(6)
# plt.plot(ypoints, color = "r", linewidth = 20) # rote, fette Linie
# plt.show()

y1 = np.array([3, 8, 1, 10])
y2 = np.array([6, 2, 7, 11])
plt.subplot(2,1,1)
plt.plot(y1)
plt.plot(y2)
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':10}
font3 = {'family':'serif','color':'green','size':15}
plt.xlabel("x-Achse", fontdict = font1) # label x
plt.ylabel("y-Achse", fontdict = font2) # label y
plt.title("Funktion", fontdict = font3, loc = "left") # title links
plt.grid(axis="y", color="red", ls = ":", linewidth = 2) # add grid
plt.subplot(2,1,2) # subplot (row,column,nr. of plot)
plt.plot(y1)
plt.plot(y2)
plt.show() # mehrere Linien
plt.figure(1)

plt.figure(2)
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 1)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 2)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 3)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 4)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 5)
plt.plot(x,y)
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 6)
plt.plot(x,y)
plt.suptitle("ALL") # übergeordneter Titel
plt.show()

plt.figure(3)
plt.scatter(x,y,color="red") # scatter plot
plt.show()

plt.figure(4)
c = ["orange", "purple", "magenta", "gray"]
plt.scatter(x,y,color=c) # verschiedene Farben
plt.show()

plt.figure(5)
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
plt.scatter(x, y, c=colors, s=colors, cmap='viridis', alpha=0.3) # mit colormapping und sizes
plt.colorbar()
plt.show()
# weitere Optionen:     alpha = Transparenz

plt.figure(6)
x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])
plt.bar(x,y) # bar graph (horizontal: barh, width,)
plt.show()

plt.figure(7)
x = np.random.normal(170, 10, 250)
print(x)
plt.hist(x) # histogramm
plt.show() 

# pie-chart = pie(y)