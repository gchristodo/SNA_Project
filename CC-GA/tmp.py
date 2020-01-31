import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
#for notebook remove below  #
#%pylab inline

#@uthor sidirasg


#neiborlist`
def nebigorList(G,no):
    allne=list(G.neighbors(no))
    return allne


#Take the neibor list @DataFrame for each nodeq
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
    # True labels of the group each student (node) unded up in. Found via the original paper
#we have 2 clusters one with node 0 a
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]




# now we have to find the maximuc cc for each nebor
#and
def ccNodeVal(g,cc):
    nodeL=[]
    km=[]
    for c in cc['neighbors']:
        k=0
        ko=0
        node1=''
        tempko =[]
        for i in c:

            k = (nx.clustering(g, i))
            print('node', c, ' has neibors ', i,' and the cc is ',k)

            if (k>ko):
                ko=k
                node1=i
            elif (k==ko):
                tempko.append(i)
        #random choose a neibor if the have the maximu cc then choose random a node
        if len(tempko)>0:
            node1=random.choice(tempko)
         #these prints are only for debuging to notify the values

        print(" The maximou CC is node",node1, 'with maximum cc ',ko,tempko)

        nodeL.append(node1)
        km.append(ko)

    #Appent To Dataset the CC node and cc Value
    #locus-based adjacency representation we have node and mmCCNode and is 'supergen' with high cc value mod1
    cc['mmCCNode']=nodeL
    cc['mmCCval']=km

clusercoeff = pd.DataFrame()
def ccnode(G):
    clusercoeff = pd.DataFrame()
    nodl=[]
    ccl=[]

    for ni in nodes:
        nodl.append(ni)
        ccl.append((nx.clustering(G,ni)))
    clusercoeff['node']=nodl
    clusercoeff['cc']=ccl
    return clusercoeff
def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst)-1, 2)}
    return res_dct



#import deep
#This is the cross over function pick a random parant
def Ucross(par1,par2):
    out = []
    for i in range (0,len(par1)):#TODO need try cach in case the length of par1!=par2
        condi=bool(random.getrandbits(1)) #condi random Out True or false
        if condi==True:
            out.append(par1[i])
        else:
            out.append(par2[i])
    return out


#take alal the neiboras of the graph to
cc=neb(G)

#run and append @Dataset for each node the max node CC with the value CC
ccNodeVal(G,cc)

#Show @Dataframe cc we have the neibors list and te cc

#graph node and mmCCnode
#graph node and mmCCnode
G2 = nx.from_pandas_edgelist(cc, 'node', 'mmCCNode')

# find the conencted componets of the new graph and is the chromosome
#the initial popoulation with2  is concomp
concomp=list(list(nx.connected_components(G2)))

numCluster=len(concomp)

nx.draw_networkx(G2)
plt.show()

#we import quality that has the modularity function 
#communities = {node: community for community, node in enumerate(neighbors.keys())}
from quality import modularity

#modularity for all chromosume
mod1=modularity(G2,concomp)
# now we have to find the number of comminites
#as intitial popoylation we have numcom



#Now we have tat mod1 the modulartity of the good inital chromosum that is going to be the first and second generation so we pot in popoulation initial
popoulationInit= pd.DataFrame()
evol= pd.DataFrame()
evol['chromB']=mod1
popoulationInit['node']=cc['node']
popoulationInit['chromB']=cc['mmCCNode']




#Random population take only the naibors in a dataframe var=500 chromosoms for each node
#we will use this function to produce random papulation, only if they are  are neibors
#maybe we havw rto examine the




chromosom=[]

loopy=100
Ga=[]
concompA=[]
while loopy>0:
    for i in cc['neighbors']:
        inde = random.randint(0, len(i) - 1)
        chromosom.append(i[inde])
    tempna='chromosom'+str(loopy)
    popoulationInit[tempna] = chromosom
    chromosom=[]
    name='G'+str(tempna)
    Ga.append(nx.from_pandas_edgelist(popoulationInit, 'node', tempna))
    #concompA.append(list(list(nx.connected_components(G3+tempna))))
    #'G'+str(tempna) = nx.from_pandas_edgelist(popoulation, 'node', tempna)
    #concomp+tempna=list(list(nx.connected_components(G3+tempna)))
    loopy = loopy - 1

#

#we create 1st generation manual manualy with best chrom
offspiring= pd.DataFrame()
offspiring['chromosom0']=popoulationInit['chromB']
offspiring['node']=cc['node']

tmp2=[]
#for each popoulation
for i in range (1,popoulationInit.shape[1]-1):
    ran=random.randint(1, (popoulationInit.shape[1] - 2))#is the random  number
    chrRan='chromosom'+str(ran)
    nameg = 'offspring' + str(i)
    chr = 'chromosom' + str(i)
    tmp2 = (Ucross(list(popoulationInit[chr]), popoulationInit[chrRan]))
    offspiring['chromosom' + str(i)] = tmp2
    tmp2 = []


#evalouation each chromosome
#we have to bult a graph for each cromosome and find the conected comunites
#then we run modularity and storees it
concomponet=[]
result= pd.DataFrame()
gp=[]
for i in range (0,offspiring.shape[1]-2): #we have 2 extra coloum 0 coloumn and node coloumn
    chro = 'chromosom' + str(i)
    gp.append(nx.from_pandas_edgelist(offspiring, chro, 'node'))
for i in gp:
    #gr=gp[i]
    concomponet.append(modularity(i,list(list(nx.connected_components(i)))))
result['modularity']=concomponet
#take finala result the result for next generation
resultF=result[result['modularity']>0.70]

#we select the second generation all the chromosomu with modularity > 0.79
offspiringN= pd.DataFrame()

offspiringN['node']=cc['node']
indx=resultF.index.values.tolist()
for i in resultF.index.values.tolist():
    offspiringN['chromosom' + str(i)] = offspiring['chromosom' + str(i)]


#we have the generation in loop
for gloop in range (1,3):
    print(i)





