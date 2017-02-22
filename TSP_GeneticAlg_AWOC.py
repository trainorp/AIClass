"""
Created on Fri Oct 21 20:27:44 2016
@author: Patrick Trainor
@course: Artificial Intelligence
@title: Project 5

Code for embedding of figure in tk credited to: 
    http://matplotlib.org/examples/user_interfaces/embedding_in_tk.html
    
Code for labeling points on figure credited to "unknown" @
    http://stackoverflow.com/posts/5147430/revisions
"""

# Imports:
import copy
import re 
import numpy as np
import sklearn.metrics as skm
import scipy as sc
import pandas as pd

# Read TSP file:
filename='Random11.tsp'
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
pdist=skm.pairwise_distances(crds)

def permAdj(perm):
    adj=np.zeros((cities,cities),dtype=np.int)
    for i in range(cities-1):
        adj[perm[i],perm[i+1]]=1
        adj[perm[cities-1],perm[0]]=1
    return adj

# Adjacency to permutation:
def adjPerm(adj):
    perm=np.array([0],dtype=np.int)
    while len(perm)<=adj.shape[1]-1:
        perm=np.append(perm,next(j for j in range(adj.shape[1]) if adj[perm[-1],:][j]==1))
    return perm

# Cost function for permutation:
def costPerm(perm):
    return np.sum(permAdj(perm)*pdist)

# Population generation function. Generates "independent births"
# AKA external immigration
def birth(births):
    perms=[np.random.permutation(range(cities)) for i in range(births)]
    perms=map(lambda x: np.append(x,x[0]),perms)
    ages=np.zeros(len(perms),dtype=np.int)
    return [perms,ages]


class population:
    def __init__(self,cities,permList,beta):
        self.cities=cities
        self.permList=permList[0]
        self.ages=permList[1]
        self.beta=beta
    
    #Convert permutation to adjacency
    def permAdj(self):
        adjs=[]
        for perm in self.permList:
            adj=np.zeros((self.cities,self.cities),dtype=np.int)
            for i in range(self.cities-1):
                adj[perm[i],perm[i+1]]=1
                adj[perm[cities-1],perm[0]]=1
            adjs=adjs+[adj]
        return adjs
    
    # Compute all costs (each permutation)
    def cost(self):
        adjs=self.permAdj()
        costs=np.array([np.sum(adj*pdist) for adj in adjs])
        return costs
    
    # Determine minimum path cost for best permutation
    def minCost(self,nmin=1):
        costs=self.cost()
        minIdx=np.argsort(costs)
        return costs[minIdx][0:nmin], minIdx[0:nmin]
    
    # Which is permutation with minimum path cost
    # LOH fix this
    def minCostPath(self,nmin=1):
        minCosts=self.minCost(nmin)
        return [self.permList[x] for x in minCosts[1]]
    
    # Determine relative fitness for each permutation
    def relFitness(self):
        costs=self.cost()
        order=np.argsort(costs)[::-1]
        invOrder=np.argsort(order)
        sortedCosts=np.sort(costs)[::-1]
        # Empirical cumulative distribution function:
        ecdf=np.arange(1,len(sortedCosts)+1)/float(len(sortedCosts))
        return ecdf[invOrder]
    
    # Determine fitness for each permutation
    def fitness(self):
        # Take ECDF function values and generate beta distribution
        prob=sc.stats.beta.ppf(self.relFitness(),1,self.beta)
        return prob
    
    # Use beta priors to determine if a permutation is eligible
    # for matting 
    def tinder(self):
        fit=self.fitness()
        return [sc.stats.binom.rvs(n=1,p=prob) for prob in fit]

# Function for mapping path to coordinates
def pathMap(path): 
    xCrds=[]
    yCrds=[]
    for i in range(len(path)):
        xCrds=xCrds+[crds[path[i]][0,0]]
        yCrds=yCrds+[crds[path[i]][0,1]]
    return [xCrds,yCrds]

# Recombination / Mating function:
def recomb(pop):
    tinder=np.where(pop.tinder())[0] # Determine perms eligible to mate
    j=len(tinder)
    newlyweds=[]
    while j>1:
        mateIdx=np.random.choice(range(len(tinder)))
        pair1=tinder[mateIdx]
        tinder=np.delete(tinder,mateIdx)
        mateIdx=np.random.choice(range(len(tinder)))
        pair2=tinder[mateIdx]
        tinder=np.delete(tinder,mateIdx)
        pair=[pair1,pair2]
        newlyweds=newlyweds+[pair]
        j=j-2
    # Make one recombination per pair (Chinese one child policy)
    children=[]
    for couple in newlyweds:
        mom=pop.permList[couple[0]]
        dad=pop.permList[couple[1]]
        for i in range(2):
            dadCopy=dad
            # Two random crossover ends
            cross=np.sort(np.random.randint(low=1,high=cities,size=2))
            cross0=cross[0]
            cross1=cross[1]
            # First part before from mom
            child=mom[cross0:cross1]
            front=np.array([],dtype=np.int)
            back=np.array([],dtype=np.int)
            l1=len(child)
            # Then dad: 
            j=0
            while l1<len(mom)-1:
                if dadCopy[j] not in np.concatenate((front,child,back)):
                    if l1<cross1:
                        front=np.append(front,dadCopy[j])
                        dadCopy=np.delete(dadCopy,j)
                        l1+=1
                    else:
                        back=np.append(back,dadCopy[j])
                        dadCopy=np.delete(dadCopy,j)
                        l1+=1
                else:
                    j+=1
            child=np.concatenate((front,child,back))
            child=np.append(child,child[0])
            children=children+[child]
    pop.permList=np.append(pop.permList,children,axis=0)
    pop.ages=np.append(pop.ages,np.zeros(len(children),dtype=np.int))
    return pop

# Function for killing off the least fit (low fitness and old age)
def death(pop,deathP):
    fit=pop.fitness()
    deaths=int(len(fit)*deathP)
    orderFit=np.argsort(fit)[::-1]
    invOrderFit=np.argsort(orderFit)
    sortedFit=np.sort(fit)[::-1]
    # Death order is linear combination of inverse of fitness and age
    ecdfFit=np.arange(1,len(sortedFit)+1)/float(len(sortedFit))*4*np.max(pop.ages)
    ecdfFit=ecdfFit[invOrderFit]
    deathFit=pop.ages+ecdfFit
    deathOrder=np.argsort(deathFit)[::-1]
    deathOrder=deathOrder[:deaths]
    pop.permList=np.delete(pop.permList,deathOrder,axis=0)
    pop.ages=np.delete(pop.ages,deathOrder,axis=0)
    return pop

# Function for inducing pointwise mutations
def mutations(pop,mutBeta,conservation=1,unifMutRate=.075):
    # Pointwise mutation probability is inversely related to fitness:
    if conservation==1:
        mutProb=1-pop.relFitness()
        mutProb=[sc.stats.beta.ppf(mutProb[i],1,mutBeta) for i in range(len(mutProb))]
        for i in range(len(pop.permList)):
            spotMutProb=mutProb[i]
            pointMut=sc.stats.bernoulli.rvs(p=spotMutProb,size=cities).astype('bool')
            for j in range(cities):
                if pointMut[j]:
                    old=pop.permList[i][j]
                    new=np.random.choice(pop.permList[i])
                    pop.permList[i][np.where(pop.permList[i]==new)[0][0]]=old
                    pop.permList[i][j]=new
    else: # Uniform mutation probability
        for i in range(len(pop.permList)):
            for j in range(len(pop.permList[i])):
                if bool(sc.stats.binom.rvs(n=1,p=unifMutRate)):
                    old=pop.permList[i][j]
                    new=np.random.choice(pop.permList[i])
                    pop.permList[i][np.where(pop.permList[i]==new)[0][0]]=old
                    pop.permList[i][j]=new
    return pop

# Function for taking a population through a generation
def generation(births,deathP,promiscuity=True,beta=3,mutBeta=10,mutate=False,conservation=1,adults=None):
    newborns=birth(births)
    # If no adults from last generation just make newborns
    if adults is None:
        pop=population(cities,permList=newborns,beta=beta)
    else:
        adults.ages=np.array(adults.ages)+1
        pop=population(cities,permList=newborns,beta=beta)
        pop.permList=np.append(pop.permList,adults.permList,axis=0)
        pop.ages=np.append(pop.ages,adults.ages)
    if deathP>0:
        pop=death(pop,deathP=deathP)
    if mutate==True:
        pop=mutations(pop,mutBeta,1)
    if promiscuity==True: # If true then mate & recombination
        pop=recomb(pop)
    return pop

def greedNN(mcps):
    mcps2=copy.copy(mcps)
    idxJoin=np.array(np.unravel_index(mcps2.argmin(),mcps2.shape))
    mcps2[:,idxJoin[0]]=np.inf
    mcps2[:,idxJoin[1]]=np.inf
    while len(idxJoin)<cities:
        idxJoin=np.append(idxJoin,np.argmin(mcps2[idxJoin[len(idxJoin)-1],:]))
        mcps2[:,idxJoin[len(idxJoin)-1]]=np.inf
    idxJoin=np.append(idxJoin,idxJoin[0])
    return idxJoin

def twoOpt(mcps,useDist=False):
    if useDist==True:
        dist=pdist
    else:
        dist=mcps
    path=greedNN(mcps)
    path=np.delete(path,len(path)-1)
    curCost=np.sum(permAdj(path)*dist)
    split=2
    while split<len(path):
        if np.sum(permAdj(np.append(path[:split,],path[split:,][::-1]))*dist)<curCost:
            path=np.append(path[:split,],path[split:,][::-1])
            curCost=np.sum(permAdj(path)*dist)
            split=2
        else:
            split+=1
    path=np.append(path,path[0])
    return path

outputFile=open(filename+'out.txt', 'w')
outputFile.write("#Input file: "+filename+"\n")
outputFile.close()

def genExperts(nExperts,iterations,contBirths,deathP,promiscuity,beta,mutBeta,mutate,mutType):
    experts=generation(initPop,deathP,promiscuity,beta,mutBeta,mutate,conservation=mutType,adults=None)
    expertNNCost=np.array([])
    expertTwoOptCost=np.array([])
    expertTwoOpt2Cost=np.array([])
    minCosts=np.array([])
    i=1
    j=1
    while i <= iterations:
        print 'i is: '+str(i)+'\n'
        outputFile=open(filename+'out.txt','a')
        outputFile.write('nExperts:'+str(nExperts)+' i is:'+str(i)+"\n")
        outputFile.close()
        oldCost=experts.minCost(1)[0][0]
        # Ensure population size stays stable
        if len(experts.permList) > 1.1*initPop:
            deathP += .05
        if len(experts.permList) < 1.1*initPop:
            deathP -= .05
        # New generation!
        experts=generation(contBirths,deathP,promiscuity,beta,mutBeta,mutate,conservation=mutType,adults=experts)
        epsilon=experts.minCost(1)[0][0]-oldCost
        # Calculate plateau run
        if epsilon == 0:
            j += 1
        else:
            j=0
        # If plateau run above tolerance end while loop
        if j>platTolerance:
            i=iterations
        # Find minimum cost paths and aggregate experts:
        mcps=experts.minCostPath(nExperts)
        mcps=map(lambda x: permAdj(x),mcps)
        mcps=1-np.divide(np.sum(mcps,axis=0)+np.transpose(np.sum(mcps,axis=0)),float(nExperts))
        expertNN=greedNN(mcps)
        expertTwoOpt=twoOpt(mcps)
        expertTwoOpt2=twoOpt(mcps,True)
        expertNNCost=np.append(expertNNCost,costPerm(expertNN))
        expertTwoOptCost=np.append(expertTwoOptCost,costPerm(expertTwoOpt))
        expertTwoOpt2Cost=np.append(expertTwoOpt2Cost,costPerm(expertTwoOpt))
        minCosts=np.append(minCosts,experts.minCost()[0][0])
        i+=1
    return experts, mcps, expertNN, expertTwoOpt, expertTwoOpt2, pd.DataFrame(np.column_stack((expertNNCost,expertTwoOptCost,expertTwoOpt2Cost,minCosts)),columns=['NNCost', 'twoOptCost', 'twoOpt2Cost', 'minCost'])

iterations=2000
platTolerance=2000
initPop=1000
contBirths=100
deathP=.2
promiscuity=True
beta=3
mutate=True
mutBeta=10
mutType=1

expertProps=[.01,.05,.10,.20]
nExperts=[int(float(initPop)*float(expertProp)) for expertProp in expertProps]

Experts=genExperts(nExperts[0],iterations,contBirths,deathP,promiscuity,beta,mutBeta,mutate,mutType)
ExpertsDF2=Experts[5]
ExpertsDF2['expertProp']=pd.Series(expertProps[0],index=ExpertsDF2.index)

for i in range(1,4):
    Experts=genExperts(nExperts[i],iterations,contBirths,deathP,promiscuity,beta,mutBeta,mutate,mutType)
    ExpertsDF=Experts[5]
    ExpertsDF['expertProp']=pd.Series(expertProps[i],index=ExpertsDF.index)
    ExpertsDF2=pd.concat([ExpertsDF2,ExpertsDF])

ExpertsDF2.to_csv(filename+'.csv')
