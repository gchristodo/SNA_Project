import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
#for notebook remove below  #
#%pylab inline

#@uthor sidirasg


#Return list from dataframe
def DataF_ToList(frameD):
    lista=[]
    for c in frameD:
        lista.append(c)
    return lista

#neiborlist
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

#run and append @Dataset for each node the max node CC with the value CC
ccNodeVal(G,cc)

#Now we can  create the initial popoulation is the node and ccNodeValq`1

#neibors is the gen we can do matuation and new
#these are
neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}
nodes = list(G.nodes())


#graph node and mmCCnode
#graph node and mmCCnode
G2 = nx.from_pandas_edgelist(cc, 'node', 'mmCCNode')

# find the conencted componets of the new graph and is the chromosome
#the initial popoulation with2  is concomp
concomp=list(list(nx.connected_components(G2)))

numCluster=len(concomp)

nx.draw_networkx(G2)
plt.show()

#communities = {node: community for community, node in enumerate(neighbors.keys())}
from quality import modularity

#modularity for all chromosume
mod1=modularity(G2,concomp)
# now we have to find the number of comminites
#as intitial popoylation we have numcom
numcom=len(concomp)

#we fount the minimum length community
def minLenNode(Listmom):
    llmin=999999999999
    nodminn=''
    for n in range(0,len(Listmom)):
        if len(Listmom[n])<llmin:
            llmin= len(Listmom[n])
            nodminn=n
            print(len(Listmom[n]),n)
            return nodminn,llmin

#we fount the minimum length community
llmin=999999999999
nodminn=''
for n in range(0,len(concomp)):
        if len(concomp[n])<llmin:
            llmin= len(concomp[n])
            nodminn=n
            print(len(concomp[n]),n)
print(nodminn,llmin)

#the nodes we have to
nodeDist=concomp[nodminn]
#for n in nodeDist:
 #   #neibors n
  #  firndn=neighbors[n]
   # for j in

cc["true"]=y_true

#ToDo Find the parametets of Genetic

#ToDo Start the Genetic implementation


#TODO here is another example find the maximum cc with Random


#create a list with cluster  coeff for each node

#loopy how popolation chomosum we want to generate may be we have to  invistigate to genarate radom numbers with out be conected
#there is a probability to have the same community some one
popoulation= pd.DataFrame()
chromosom=[]
popoulation['node']=cc['node']
loopy=50 #initial popoulation

#Random population take only the naibors in a dataframe 500 chromosoms for each node
#we will use this function to produce random papulation, only if they are  are neibors

Ga=[]
concompA=[]
while loopy>0:
    for i in cc['neighbors']:
        inde = random.randint(0, len(i) - 1)
        chromosom.append(i[inde])
    tempna='chromosom'+str(loopy)
    popoulation[tempna] = chromosom
    chromosom=[]
    name='G'+str(tempna)
    Ga.append(nx.from_pandas_edgelist(popoulation, 'node', tempna))
    #concompA.append(list(list(nx.connected_components(G3+tempna))))
    #'G'+str(tempna) = nx.from_pandas_edgelist(popoulation, 'node', tempna)
    #concomp+tempna=list(list(nx.connected_components(G3+tempna)))
    loopy = loopy - 1

#
# for i in range (0, len(Ga)):
#     concompA.append(list(list(nx.connected_components(Ga[i]))))
#     nx.draw_networkx(Ga[i])
#     plt.show()
# for i in concompA:
#     for j in i:
#         print(len(j))


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

ccL=cc['mmCCNode']

popgen= pd.DataFrame()
gen=[]
for i in range (1,50):#here 50 i loopy
    nameg='offspring'+str(i)
    chr='chromosom'+str(i)
    gen=(Ucross(list(popoulation[chr]),list(cc['mmCCNode'])))
    popgen['gen'+str(i)]=gen
    gen=[]

popgen['node']=cc['node']
gp=[]
for i in range(1,50):
    chro='gen'+str(i)
    gp.append(nx.from_pandas_edgelist(popgen, chro, 'node'))

concomponet=[]
result= pd.DataFrame()

for i in gp:
    #gr=gp[i]
    concomponet.append(modularity(i,list(list(nx.connected_components(i)))))

result['modularity']=concomponet
#take the 3 maximum
tmp=result['modularity'].nlargest(3)
#take the 3 most modularity to next popoulation
resultF= pd.DataFrame()
for i in tmp.index.values.tolist():
    resultF['gen'+str(i)]=popgen['gen' + str(i)]

resultF['ccNode']=list(cc['mmCCNode']


