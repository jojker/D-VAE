# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:19:35 2021

@author: ponce
"""

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from scipy.optimize import fminbound
from util import load_3D_graphs, decode_igraph_to_3D

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
# %matplotlib notebook
# %matplotlib inline
# %matplotlib auto

""" 
Creating dummy 3D scene graphs

These graphs will correspond to simple scenes with a few objects that have few
properties. These are spheres that have locations, radii, and color. There are
probability distributions governing the number of spheres, the likelihood that
two spheres are connected, the radii and the colors. We then randomly generate
3D scene graphs and write them to a file. Then we test our generator. 
"""


# define node types
# types 1 & 2 are start and end by default
# 1 shape type (sphere)
# properties: X, Y, Z, R, G, B, rad
# defaults: 0, 0, 0, 0.5, 0.5, 0.5, 0.05
# thus 8 types and each weight value is naturally 0-1
# type 0==shape, 1->7 follow in order



""" 
define probability distributions 
"""
# a distrubtion over shape degrees
def seed2Deg(seed):
    # fat tailed weibull centered on 2, zero at 5 
    # weibull shape and scale = 2.34 gives approximately media at 2, max at 5, with 0.997 the exact value at 5
    # inv distribution is 
    scale=2.34
    shape=2.34
    correction=0.997
    return min(round(scale*float(-np.log(1-correction*seed))**(1./shape)),5)
    
# a dependent distribution over inheritence and type (prop | nAncestors,type)
def seed2inheritence(seed,pType,nAncestors,newTypes):
    # ptype (property type) is defined with respect to the node type
    # highly likely to get new X Y Z
    if pType in range(1,4):
        return seed < 0.7
    # less likely to get new color, increasing with nAncestors, increasing a lot if color already changed
    if pType in range(4,7):
        baseProb = 0.1+0.4*expit(nAncestors-2.)
        if len(set(newTypes) and set(range(4,7))) > 0:
            baseProb = 1-(1-baseProb)/3
        return seed < baseProb
    # radii more likely than color, decaying with nAncestors
    if pType==7:
        baseProb = 0.3-0.2*expit(nAncestors-4.)
        return seed < baseProb
    # we should never get this far
    return False
    
# a distribution over radii
def seed2Rad(seed):
    # a fat tailed weibull centered on 0.05, support eps-0.5
    scale=0.066284/2
    shape=1.3
    correction=1.
    return min(max(0.0001,scale*float(-np.log(1-correction*seed))**(1./shape)),0.5)
    
# a distribution over colors
def seed2Color(seed):
    # uniform
    return seed
    
# a distribution over X Y and Z (dependent on Ancestor locations)
    # 1-gaussian convolution at current X Y and Z
def seed2Coordinate(seed,AncestorsCoords,AncestorsRadii):
    # use a 1 D mixture of gaussians, and invert the cdf using the seed value as input
    # can only be inverted as an optimization problem
    # the most likely points are those outside the existing shells but overlap is not ruled out
    # each existing shell is a gaussian which has been "inverted" (subtracted from its maximum)
    if len(AncestorsCoords) == 0:
        return seed
    else:
        return fminbound(invGaussMix_helper,0.,1.,args=(seed,AncestorsCoords,AncestorsRadii),disp=0)

def invGaussMix_helper(val,seed,mus,sigmas):
    seedOut=0.0
    nAncestors=len(mus)
    for mu,sigma in zip(mus,sigmas):
        sigma2=sigma/2
        shift = norm.cdf(0,loc=mu,scale=sigma2)
        renormalize = ( norm.pdf(mu,loc=mu,scale=sigma2)  -  norm.cdf(1,loc=mu,scale=sigma2)-shift ) 
        seedOut +=  ( val*norm.pdf(mu,loc=mu,scale=sigma2)  -  (norm.cdf(val,loc=mu,scale=sigma2)-shift) ) / renormalize
    return abs( (seedOut / nAncestors) - seed )

    
# A recursive function to handle assigning children and properties          
def addNodesWChildren(row,nAncestors,parentNode):
    # get children for this child node
    if nAncestors < 4 and len(row) < 15:
        seed=np.random.rand()
        nChildren = seed2Deg(seed)
    else:
        nChildren = 0
    # step through this node and it's children        
    for childNdx in range(1+nChildren):
        nNodes=len(row)-7
        node=[0]
        if parentNode is None:
            propConnections=np.random.rand(7).tolist()
        else:
            propConnections=[0] * 7
            propNdcs = np.random.permutation(list(range(0,7))).tolist()
            newTypes=[];
            for propNdx in propNdcs:
                seed=np.random.rand()
                if seed2inheritence(seed,propNdx+1,nAncestors,newTypes):
                    propConnections[propNdx] = np.random.rand()
                    newTypes.append(propNdx+1)
        propConnections=propConnections+ [0]*nNodes
        node = node + propConnections
        # connect the child to the parent
        if parentNode is not None:
            node[parentNode] = np.random.rand()
        
        if len(node)>8 and 0.0 in node[1:8] and all([x == 0.0 for x in node[8:]]):
            print('I need debugging')
            
        if childNdx != 0: # store the child node and give it children
            row.append(node)
            row = addNodesWChildren(row,nAncestors+1,parentNode)
        else: # store the initializing node 
            row.append(node)
            parentNode = len(row)
    return row    


# step through a large number of potential graphs

nGraphs=2000
graphs2Write=[]
f=open('data/Test3DGraphs.txt', 'w')
for gNdx in range(nGraphs):
    # generate the graphs
    row=[]
    # first 7 nodes are properties
    for propNdx in range(0,7):
        # pad with increasing number of zeros to indicate no connections between property nodes
        node = [propNdx+1]
        node  = node + ([0] * propNdx)
        row.append(node)
    
    # now we assign children
    allNodesAssigned=False
    # keep assigning root nodes until we have at least 5 total nodes
    while not allNodesAssigned:
        row=addNodesWChildren(row,0,None)
        nNodes = len(row)-7
        allNodesAssigned = nNodes > 4
                
    # convert to string
    rowString ='[' + ','.join(str(x) for x in row) + ']\n'
    # write to file
    f.write(rowString)
f.close()

print('all scene graphs assigned')


# step through a row and assign properties

# step through levels in the hierarchy first
# get all root nodes and gather their children and compile the properties
# iterate through the root nodes assigning properties to one child at a time (because of how locations are set)

        
# define recursive function to assign properties
# inherit properties from parent node to child node while growing list     
def propertyCollection(v,g,initProperties):
    childNodeNdcs=[e.target for e in v.all_edges() if e.source==v.index and g.vs[e.target]['type']==2 ]
    propertyEdges=[e for e in v.all_edges() if e.target==v.index and g.vs[e.source]['type'] in list(range(3,10))]
    properties=initProperties.copy()
    for e in propertyEdges:
        properties[e.source-3]=e['seed']
    return properties, childNodeNdcs
    
def childAccumulation(childNodeNdcs,g,properties):
    propertyList=list()
    childList=list()
    for childNdx in childNodeNdcs:
        v=g.vs[childNdx]
        (propertiesTemp, childNodeNdcsTemp) = propertyCollection(v,g,properties)
        propertyList.append(propertiesTemp)
        childList.append(childNodeNdcsTemp)
    return propertyList, childList

if False:
    g_list=load_3D_graphs('Test3DGraphs',n_types=8,fmt = 'igraph')
    scene_list=[];
    
    for g in list(zip(*g_list[0]))[0]:
        rootNodeTest = [ all([g.vs[e.source]['type']-2 in list(range(1,8)) for e in v.all_edges() if e.target == v.index ] ) and v['type']==2  for v in g.vs()]
        rootNodes = [i for i, x in enumerate(rootNodeTest) if x]
        
        nodeProperties = [[[],[]]] * len(rootNodes) # first is list of child property seed values, second are the actual values
        # gather the seed values with all nodes at one hierarchy level adjacent to each other
        for ndx, rootNdx in enumerate(rootNodes):
            v=g.vs[rootNdx]
            (properties, childNodeNdcs) = propertyCollection(v,g,[0] * 7)
            childList=[childNodeNdcs]
            propertyList=[[properties]]
            cnt=0
            while not all(len(x)==0 for x in childList):
                cnt += 1
                # now step through and add child node properties to the list
                # then add the properties of their children, etc
                oldChildList=childList.copy()
                childList=list()
                lastPropNdx=len(propertyList)-1
                propertyList.append([])
                for ii, childNodeNdcs in enumerate(oldChildList):
                    (propertyListTemp, childListTemp) = childAccumulation(childNodeNdcs,g,propertyList[lastPropNdx][ii])
                    propertyList[-1]=propertyList[-1]+propertyListTemp
                    childList=childList+childListTemp
            nodeProperties[ndx][0]=propertyList
        
        # now gather the actual property values
        lastH = [0] * len(rootNodes) 
        complete = [False] * len(rootNodes)
        AncestorsCoords = [[],[],[]]
        AncestorsRadii = []
        while not all(complete):
            for ndx in range(len(nodeProperties)): # step through root nodes
                if lastH[ndx]<len(nodeProperties[ndx][0]):
                    if lastH[ndx] == 0: # the value list has not been allocated yet
                        nodeProperties[ndx][1]=[[]] * len(nodeProperties[ndx][0])
                    propertyList=nodeProperties[ndx][0][lastH[ndx]]
                    nodeProperties[ndx][1][lastH[ndx]]=[[0] * 7 ] * len(nodeProperties[ndx][0][lastH[ndx]])
                    for ndx2, properties in enumerate(propertyList):
                        for ptype, seed in enumerate(properties):
                            if ptype<3: # get the x y z positions and store them
                                val=seed2Coordinate(seed,AncestorsCoords[ptype],AncestorsRadii)
                                AncestorsCoords[ptype].append(val)
                            elif ptype in range(3,6): # get the r g b values and store them
                                val=seed2Color(seed)
                            elif ptype==6: # get the radius
                                val=seed2Rad(seed)
                                AncestorsRadii.append(val)
                            nodeProperties[ndx][1][lastH[ndx]][ndx2][ptype]=val
                    lastH[ndx]=lastH[ndx]+1
                else:
                    complete[ndx]=True
        scene_list.append(nodeProperties)
                    
            
    
    # create a base sphere to use for modification later
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
        
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    g2Plot=np.random.randint(0,high=len(scene_list),size=1)[0]
    print(g2Plot)
    objList=scene_list[g2Plot];
    for rootObj in objList:
        for objParts in rootObj:
            for valueList in objParts:
                for values in valueList:
                    x2=values[0] + values[6] * x
                    y2=values[1] + values[6] * y
                    z2=values[2] + values[6] * z
                    rgb=values[3:6]
                    print(rgb)
                    
                    surf = ax.plot_surface(x2, y2, z2, color = rgb,
                                           linewidth=0, antialiased=True)
                    ax.axes.set_xlim3d(left=-1, right=2) 
                    ax.axes.set_ylim3d(bottom=-1, top=2) 
                    ax.axes.set_zlim3d(bottom=-1, top=2) 
    plt.axis('off')
    plt.show()        
