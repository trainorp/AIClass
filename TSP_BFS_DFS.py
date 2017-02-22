"""
Created on Thu Sep 8 14:37:34 2016

@author: Patrick Trainor
@course: Artificial Intelligence
@title: Project 2

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

# Create Tk object:
root=Tk.Tk()
root.wm_title("BFS and DFS Search")

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

#Add adjacency matrix:
adj=np.array([[0,1,1,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0],
               [0,0,0,1,1,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0,0],
               [0,0,0,0,0,0,1,1,0,0,0],[0,0,0,0,0,0,0,1,0,0,0],
               [0,0,0,0,0,0,0,0,1,1,0],[0,0,0,0,0,0,0,0,1,1,1],
               [0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,1]],dtype="bool")

#Breadth first algorithm:
def bfs(adj,start,goal):
    vertex=start
    goalAcheived=vertex==goal
    visit=[vertex]
    edges=[]
    while goalAcheived is False:
        neighbors=np.where(adj[vertex,:])[0].tolist() #Find neighbors
        for neighbor in neighbors:
            visit=visit+[neighbor] #visit neighbors
            edges=edges+[[vertex,neighbor]] #Edges to neighbor
            goalAcheived=neighbor==goal #Check is neighbor goal?
            if goalAcheived: #If neighbor is goal then stop
                break
        visit=[x for x in visit if x!=vertex] #Remove city from queue
        vertex=visit[0] #Choose next city in queue
    path=[edges.pop()] #Add edge to path
    while path[0][0]!=start: #Backtrace
        path=[[x for x in edges if x[1]==path[0][0]][0]]+path
    return path

#Depth first algorithm 
def dfs(adj,start,goal):
    nextVertex=vertex=start
    goalAcheived=vertex==goal
    visit=[]
    edges=[]
    while goalAcheived is False:
        vertex=nextVertex
        neighbors=np.where(adj[vertex,:])[0].tolist() #Find neighbors
        if neighbors==[]: #Iff no more neighbors in stack go back
            while neighbors==[]:
                vertex=visit.pop()
                neighbors=np.where(adj[vertex,:])[0].tolist()
        visit=visit+neighbors #Add new neighbors to stack
        nextVertex=visit.pop() #Next city is last in stack
        edges=edges+[[vertex,nextVertex]] #add edges
        goalAcheived=edges[-1][-1]==goal #Check goal city?
    path=[edges.pop()] #Add to path
    while path[0][0]!=start: #Backtrace
        path=[[x for x in edges if x[1]==path[0][0]][0]]+path
    return path
    
def pathMap(path): #Function for mapping path to coordinates
    xCrds=[]
    yCrds=[]
    for i in range(len(path)):
        xCrds=xCrds+[[crds[path[i][0]][0,0],crds[path[i][1]][0,0]]]
        yCrds=yCrds+[[crds[path[i][0]][0,1],crds[path[i][1]][0,1]]]
    return([xCrds,yCrds])

#Execute BFS
t0=time.time()
bfsPath=bfs(adj,0,10)
t1=time.time()
bfsTime=t1-t0
bfsCrds=pathMap(bfsPath)

#Execute DFS
t0=time.time()
dfsPath=dfs(adj,0,10)
t1=time.time()
dfsTime=t1-t0
dfsCrds=pathMap(dfsPath)

#Determine the cities and edges of the whole 
xCrds=[]
yCrds=[]
for i in range(10):
    for j in range(11):
        if adj[i,j]:
            xCrds=xCrds+np.squeeze(crds[:,0][[i,j]]).tolist()
            yCrds=yCrds+np.squeeze(crds[:,1][[i,j]]).tolist()

#Function for plotting cities, edges, and paths:
def plotFun(xCrds,yCrds,pathCrds=[],plotPath=False):
    fig=plt.figure(figsize=(5, 4), dpi=100)
    f1=fig.add_subplot(111)
    f1.plot(crds[:,0],crds[:,1],'ro')
    f2=fig.add_subplot(111)
    for i in range(len(xCrds)):
        f2.plot(xCrds[i],yCrds[i],'--',color='c')
    if plotPath:
        f3=fig.add_subplot(111)
        for i in range(len(pathCrds[0])):
            f3.plot(pathCrds[0][i],pathCrds[1][i],'-',color='r')
    f4=fig.add_subplot(111)
    labs=map(str,range(11))
    for label, x, y in zip(labs,crds[:,0],crds[:,1]):
        f4.annotate(label, xy = (x, y),xytext = (-10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    return fig


f=plotFun(xCrds,yCrds,bfsCrds,plotPath=True)
f2=plotFun(xCrds,yCrds,dfsCrds,plotPath=True)

#Add listbox with results to tk window
listbox = Tk.Listbox(root)
listbox.pack(side=Tk.TOP, fill=Tk.X)
listbox.insert("end","BFS Path: "+str(bfsPath))
listbox.insert("end","DFS Path: " +str(dfsPath))

#Add figure to tk window
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
canvas = FigureCanvasTkAgg(f2, master=root)
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

#Execute tk main loop
Tk.mainloop()

#Write results to file
with open(filename.split('.')[0]+'Solution.txt','a') as tf:
    tf.write("Input file: "+filename)
    tf.write("\n")
    tf.write("BFS Path: " +str(bfsPath))
    tf.write("\n")
    tf.write("DFS Path: "+str(dfsPath))
    tf.write("\n")
    tf.write("BFS time: "+str(bfsTime)+" DFS time: "+str(dfsTime))
    tf.close()
