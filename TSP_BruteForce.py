"""
Created on Thu Aug 25 20:37:34 2016

@author: Patrick Trainor
@course: CECS 545 AI

Code for embedding of figure in tk credited to: 
    http://matplotlib.org/examples/user_interfaces/embedding_in_tk.html
    
Code for labeling points on figure credited to "unknown" @
    http://stackoverflow.com/posts/5147430/revisions
"""

# Imports:
import time
import numpy as np
import re 
import sklearn.metrics.pairwise as prw
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from itertools import permutations
import sys
import Tkinter as Tk

#Start the clock
t0=time.time()

# Create Tk object:
root=Tk.Tk()
root.wm_title("TSP Solution")

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

# Determine possible paths by generating permutations
# City is fixed "0" so each is incremented and zeros concatenated
perms=list(permutations(range(cities-1),cities-1))
perms=np.asarray(perms)+1
permZeros=np.zeros((len(perms[:,0]),1),dtype=np.int)
perms=np.concatenate((permZeros,perms),axis=1)

# Initialize cost matrix and path iteration; compute cost
pathCost=float('inf')
bestPath=perms[0]
for perm in perms:
    pathCostNew=0
    for i in (range(cities-1)):
        pathCostNew = pathCostNew + pdist[list(perm)[i],list(perm)[i+1]]
    pathCostNew = pathCostNew + pdist[list(perm)[i+1],list(perm)[0]]
    if pathCostNew < pathCost: #Retain path only if best so far
        pathCost=pathCostNew
        bestPath=perm

#Determine best path and coordinates of best path
bestDist=pathCost
pathCoords=crds[[i for i in bestPath],:]
pathCoords=np.concatenate((pathCoords,pathCoords[0,:]),0)

#Plot coordinates
f=Figure(figsize=(5, 4), dpi=100)
f1=f.add_subplot(111)
f1.plot(crds[:,0],crds[:,1],'ro')
f2=f.add_subplot(111)
f2.plot(pathCoords[:,0],pathCoords[:,1])
f3=f.add_subplot(111)

#Label coordinates on plot
labs=map(str,list(bestPath))
for label, x, y in zip(labs,pathCoords[:,0],pathCoords[:,1]):
    f3.annotate(label, xy = (x, y),xytext = (-10, 10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

#Add listbox with results to tk window
listbox = Tk.Listbox(root)
listbox.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
listbox.insert("end", "Best Path: "+str(bestPath))
listbox.insert("end","Distance: " +str(bestDist))

#Add figure to tk window
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

#Add toolbar to tk window
toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

#click event handler
def on_key_event(event):
    print('you pressed %s' % event.key)
    key_press_handler(event, canvas, toolbar)

canvas.mpl_connect('key_press_event', on_key_event)

#Quit event handler
def _quit():
    root.quit()
    root.destroy() 
button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)

#Stop the clock
t1=time.time()

#Execute tk main loop
Tk.mainloop()

#Write results to file
with open(filename.split('.')[0]+'Solution.txt','a') as tf:
    tf.write("Input file: "+filename)
    tf.write("\n")
    tf.write("Distance: " +str(bestDist))
    tf.write("\n")
    tf.write("Best Path: "+str(bestPath))
    tf.write("\n")
    tf.write("Time elapsed: "+str(t1-t0))
    tf.close()
