import numpy as np
import pandas as pd
import networkx as nx
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance
import pandas as pd






#neiborlist
def nebigorList(G,no):
    allne=list(G.neighbors(no))
    return allne
#Take the neibor list @DataFrame for each node
def neb(z):
    nodeneibdf = pd.DataFrame()
    a=[]
    n=[]
    for i in z:
        g = nebigorList(z,i)
        a.append(g)
        n.append(i)
    nodeneibdf['node']=n
    nodeneibdf['neighbors']=a
    return nodeneibdf





G = nx.karate_club_graph()
pos = nx.spring_layout(G)


#take all the nodes of graph
nodeAll=[]

#take alal the neiboras of the graph to
cc=neb(G)

# now we have to find the maximuc cc for each nebor
#and
def ccNodeVal(g,cc):
    nodeL=[]
    km=[]
    for c in cc['neighbors']:
        k=0
        ko=0
        node1=''
        for i in c:
            k = (nx.clustering(g, i))
            print('node', c, ' has neibors ', i,' and the cc is ',k)

            if (k>ko):
                ko=k
                node1=i
        print("The maximou CC is node",node1, 'with maximum cc ',ko)
        nodeL.append(node1)
        km.append(ko)

    #Appent To Dataset the CC node and cc Value
    cc['mmCCNode']=nodeL
    cc['mmCCval']=km

ccNodeVal(G,cc)

#ToDo Now we can  create the initial popoulation is the node and ccNode


