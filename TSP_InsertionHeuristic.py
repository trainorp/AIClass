"""
Created on Sun Sep 18 20:37:53 2016
@author: Patrick Trainor
@course: Artificial Intelligence
@title: Project 3

Code for embedding of figure in tk credited to: 
    http://matplotlib.org/examples/user_interfaces/embedding_in_tk.html
    
Code for labeling points on figure credited to "unknown" @
    http://stackoverflow.com/posts/5147430/revisions
"""

# Imports:
import time
import pandas as pd
import numpy as np
import re 
from math import sqrt
import sklearn.metrics.pairwise as prw
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import sys
import Tkinter as Tk

# Get filename from system args
filename = sys.argv[-1]
tsp_file=open(filename)

# Open and read the file lines
tsp_read=tsp_file.read().splitlines()
tsp_file.close()

# Find the number of cities
for line in tsp_read:
    if line.startswith('DIMENSION'):
        cities=int(re.findall(r'\d+', line)[0])

# Find the line of the file in which coordinates start
start_line=tsp_read.index('NODE_COORD_SECTION')+1

# Create matrix of pairwise distances
crds=[str.split(line) for line in tsp_read[start_line:(start_line+cities)]]
crds=np.matrix(crds).astype(np.float)[:,1:] 
pdist=prw.pairwise_distances(crds)
np.fill_diagonal(pdist,float('inf'))
pdist=pd.DataFrame(pdist)

# First path:
edges=[]
tour=[]
outCities=range(cities)
city1=pdist.min(axis=1).idxmin() #Start city
city2=pdist.loc[city1].idxmin() #End city
tour=tour+[city1,city2]
tourOfTours=[tour]
outCities=[x for x in outCities if x not in tour] # Remove from queue

# Line distance function:
def lineDist(city1,city2,city0):
    return abs((crds[city2,0]-crds[city1,0])*(crds[city1,1]-crds[city0,1])-(crds[city1,0]-crds[city0,0])*(crds[city2,1]-crds[city1,1]))/sqrt((crds[city2,0]-crds[city1,0])**2+(crds[city2,1]-crds[city1,1])**2)

while len(outCities)>0:
    # For loop for adding vertices with lowest distance to current edges in tour
    insertPos=int() # Position in tour where vertex will be added
    minLineDist=float('inf')
    for i in range(len(tour)-1): # Loop over edges currently in the tour
        lineDists=[lineDist(tour[i],tour[i+1],x) for x in outCities]
        minLineDistCurrent=min(lineDists)
        if minLineDistCurrent<minLineDist:
            addCity=outCities[lineDists.index(min(lineDists))]
            insertPos=i+1
            minLineDist=minLineDistCurrent
    tour=tour[0:insertPos]+[addCity]+tour[insertPos:] # Add out vertex with least distance
    tourOfTours=tourOfTours+[tour]
    outCities=[x for x in outCities if x not in tour] # Remove vertex from queue

def pathMap(path): #Function for mapping path to coordinates
    xCrds=[]
    yCrds=[]
    for i in range(len(path)):
        xCrds=xCrds+[crds[path[i]][0,0]]
        yCrds=yCrds+[crds[path[i]][0,1]]
    return [xCrds,yCrds]

pathCrds=[pathMap(path) for path in tourOfTours]

def pathCost(path):
    cost=0
    for i in range(len(path)-1):
        cost=cost+pdist.loc[path[i]].loc[path[i-1]]
    return cost

pathCosts=[pathCost(path) for path in tourOfTours]

def plotFun(i):
    xScale=(crds.max(0)[0,0]-crds.min(0)[0,0])*.2
    xRange=[crds.min(0)[0,0]-xScale,crds.max(0)[0,0]+xScale]
    yScale=(crds.max(0)[0,1]-crds.min(0)[0,1])*.2
    yRange=[crds.min(0)[0,1]-yScale,crds.max(0)[0,1]+yScale]
    fig=plt.figure(figsize=(5,4),dpi=100)
    f1=fig.add_subplot(111)
    f1.plot(crds[:,0],crds[:,1],'ro')
    plt.ylim(yRange)
    plt.xlim(xRange)
    f2=fig.add_subplot(111)
    f2.plot(pathCrds[i][0],pathCrds[i][1],'--',color='c')
    f4=fig.add_subplot(111)
    labs=map(str,range(11))
    for label, x, y in zip(labs,crds[:,0],crds[:,1]):
        f4.annotate(label, xy = (x, y),xytext = (-15, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    return fig

# Create Tk object:
root=Tk.Tk()
root.wm_title("TSP Solution")

#Add figure to tk window
fig=plotFun(0)
canvas=FigureCanvasTkAgg(fig,master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

# Listbox with cost:
listbox = Tk.Listbox(root)
listbox.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
listbox.insert("end", "Cost: "+str(pathCosts[0]))

#Add toolbar to tk window
toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

# Next insertion button event handler
i=1
def idk():
    global i
    global canvas
    global button2
    global listbox
    if i < len(pathCrds)-1:
        canvas.get_tk_widget().destroy()
        fig=plotFun(i)
        canvas=FigureCanvasTkAgg(fig,master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        listbox.destroy()
        listbox = Tk.Listbox(root)
        listbox.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        listbox.insert("end", "Cost: "+str(pathCosts[i]))
        i+=1
    if i == len(pathCrds)-1:
        canvas.get_tk_widget().destroy()
        fig=plotFun(i)
        canvas=FigureCanvasTkAgg(fig,master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        listbox.destroy()
        listbox = Tk.Listbox(root)
        listbox.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        listbox.insert("end", "Cost: "+str(pathCosts[i]))
        button2.destroy()

#Quit event handler
def _quit():
    root.quit()
    root.destroy() 

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)
button2 = Tk.Button(master=root, text='Next Insertion', command=idk)
button2.pack(side=Tk.BOTTOM)

#Execute tk main loop
Tk.mainloop()

with open(filename.split('.')[0]+'Solution.txt','a') as tf:
    tf.write("Input file: "+filename)
    tf.write("\n")
    tf.write("Paths with insertions and cost: ")
    tf.write("\n")
    for i in range(len(tourOfTours)):
        tf.write(str(tourOfTours[i])+" Cost: "+str(pathCosts[i]))
        tf.write("\n")
    tf.close()