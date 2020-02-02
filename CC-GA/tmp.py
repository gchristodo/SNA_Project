import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#for notebook remove below  #
#%pylab inline

#@uthor sidirasg


#neiborlist`
def nebigorList(G,no):
    allne=list(G.neighbors(no))
    return allne

#mutation replace a comunity random with other

def mutation(chromosum,nodes):
    nl=list(nodes)
    random.shuffle(nl)  #take randomly one node
    rplc=nl.pop()
    indxrp= random.randint(0,len(chromosum))
    chromosum[indxrp] =rplc #replace the random node with the  random aother node
    return chromosum



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


mutationRate=0.2
prob=0.8 #prob of croos over

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
           # print('node', c, ' has neibors ', i,' and the cc is ',k)

            if (k>ko):
                ko=k
                node1=i
            elif (k==ko):
                tempko.append(i)
        #random choose a neibor if the have the maximu cc then choose random a node
        if len(tempko)>0:
            node1=random.choice(tempko)
         #these prints are only for debuging to notify the values

        #print(" The maximou CC is node",node1, 'with maximum cc ',ko,tempko)

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
    for i in range (0,len(par1)):#TODOneed try cach in case the length of par1!=par2
        condi=bool(random.getrandbits(1)) #condi random Out True or false
        if condi==True:
            out.append(par1[i])
        else:
            out.append(par2[i])
    return out

#all nodes
nodesA=list(nx.nodes(G))


########################################################################################################################
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


#######################################################G2
#disable the comment to show the local briges from dataset FROM Υthe graph.. if we plot the graph we see the communites apears  and the eages are missinf is local briges
#very nice method to found local briges++
#nx.draw_networkx(G2)
#plt.show()

#we import quality that has the modularity function 
#communities = {node: community for community, node in enumerate(neighbors.keys())}
from quality import modularity

#modularity for all chromosume
mod1=modularity(G2,concomp)
# now we have to find the number of comminites
#as intitial popoylation we have numcom



#Now we have tat mod1 the modulartity of the good inital chromosum that is going to be the first and second generation so we pot in popoulation initial
popoulationInit= pd.DataFrame()
#evol= pd.DataFrame()
#evol['chromB']=mod1
popoulationInit['node']=cc['node']
popoulationInit['chromB']=cc['mmCCNode']




#Random population take only the naibors in a dataframe var=500 chromosoms for each node
#we will use this function to produce random papulation, only if they are  are neibors
#maybe we havw rto examine the




chromosom=[]

loopy=10000
neibrLoop=int(0.5*loopy)
chromosomN=int(0.5*loopy) #for indexing name
randomn=int(loopy-neibrLoop)

Ga=[]
concompA=[]
#create fist initail popoulation where there are neibors the 50% of initial population
chromosom=[]




while neibrLoop>0:
    for i in cc['neighbors']:
        k = list(i)
        random.shuffle(k)
        chromosom.append(k.pop())
    tempna = 'chromosom' + str(neibrLoop)
    popoulationInit[tempna] = chromosom
    chromosom = []
    #name = 'G' + str(tempna)
    neibrLoop = neibrLoop - 1


    #----
    # for i in cc['neighbors']:
    #     #inde = random.randint(0, len(i) - 1) #does not work very wall onother random we need
    #     chromosom.append(i[inde])
    # tempna='chromosom'+str(neibrLoop)
    # popoulationInit[tempna] = chromosom
    # chromosom=[]
    # name='G'+str(tempna)


    #---------------
    #Ga.append(nx.from_pandas_edgelist(popoulationInit, 'node', tempna))
    #concompA.append(list(list(nx.connected_components(G3+tempna))))
    #'G'+str(tempna) = nx.from_pandas_edgelist(popoulation, 'node', tempna)
    #concomp+tempna=list(list(nx.connected_components(G3+tempna)))


#create random popoulation of the graph //the nodes could belong in the same  community even if they are not neaibors

# random the nodes @nodesB for each nodeA create a link



#we have to be carefiulk with loopy index because is the index of chromosum and we have deiferent names with other index
chromosom=[]
while randomn<=loopy:
    nodesB = list(nx.nodes(G))
    random.shuffle(nodesB)#random select a node
    chromosom = []
    for i in nodesA:
        tempna = 'chromosom' + str(randomn+1)# add 3 bacouse we apend the node and chromosomu0
        chromosom.append(nodesB.pop())
    popoulationInit[tempna] = chromosom
    chromosom = []
    randomn = randomn +1


    

# while randomn>0:
#     for i in cc['neighbors']:
#         inde = random.randint(0, len(i) - 1)
#         chromosom.append(i[inde])
#     tempna='chromosom'+str(loopy)
#     popoulationInit[tempna] = chromosom
#     chromosom=[]
#     name='G'+str(tempna)
#     Ga.append(nx.from_pandas_edgelist(popoulationInit, 'node', tempna))



    #concompA.append(list(list(nx.connected_components(G3+tempna))))
    #'G'+str(tempna) = nx.from_pandas_edgelist(popoulation, 'node', tempna)
    #concomp+tempna=list(list(nx.connected_components(G3+tempna)))
   # randomn = randomn - 1



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
result['grpah']=gp
#take finala result the result for next generation
resultF=result[result['modularity']>0.5]

#we select the second generation all the chromosomu with modularity > 0.79
offspiringN= pd.DataFrame()

offspiringN['node']=cc['node']
indx=resultF.index.values.tolist()
#indx is index for offspring
bbf= pd.DataFrame()
bbf['node']=cc['node']
for i in resultF.index.values.tolist():
    offspiringN['chromosom' + str(i)] = offspiring['chromosom' + str(i)]

#take the 3 best modularity to nxt generation with oute
bb=resultF.sort_values(by=['modularity'],ascending=False).head(3)
for b in bb.index.values.tolist():
    bbf['chromosom'+str(b)]=offspiring['chromosom'+str(b)]
bbf['chromosom0']=offspiring['chromosom0']
#bbf dataframe is the best chromeseme for the generation -- servining the evolou natural choise so
#next generation Offsring N and bbf the Bad dataframe just ignore
#offspiringNf= pd.DataFrame() ##Απευθεας αναθεση next generation each generation remove the 3 worst and rteplace with the 3 best
#so when we are finish in this dataframe will we have the soloution
#for i in bbf:
 #   offspiringNf['chromosom' + str(i)] = offspiring['chromosom' + str(i)]
offspiring= pd.DataFrame() #delete the first cross over
print("Genereration")
#Above was initialization papoulation Below will be next generations  tha range of the
#ofsringn
#popoulationInit=pd.DataFrame()
#we have the generation in loop from now each generation will generated in loop  wile we do not have impovments
######################################LOOP######################################################################################################################################
for generation in range (1,100):
    indx2=[]
    gpdf = pd.DataFrame()

    tmp2=[]
    #for i in range(0, offspiringN.shape[1] - 1):
    offspiring = pd.DataFrame()
    offspiring['node'] = cc['node']

    for ind in indx:
        if isinstance(indx[0], int):  #we have different structurs when the inex is oiut of the loop from inital population
            #offspiring['node']=cc['node']
            #TODO we have to add the probability of cross
            ran =  random.choice(indx)  # is the random  chose from the list
            chrRan = 'chromosom' + str(ran)
            off = 'chromosom' + str(ind)
            #probability to cross
            rando=random.randint(1, 100)
            perh=prob*100
            if perh>rando:
                tmp2 = (Ucross(list(offspiringN[off]), offspiringN[chrRan]))
                offspiring['chromosom' + str(ind)] = tmp2
                indx2.append('chromosom' + str(ind))
                tmp2=[]
        else:
            ##############################
             #   offspiring['node'] = cc['node']
                # TODO we have to add the probability of cross
                ran = random.choice(indx)  # is the random  chose from the list
                chrRan = str(ran)
                off = str(ind)
            # probability to cross
                rando = random.randint(1, 100)
                perh = prob * 100
                if perh > rando:
                    tmp2 = (Ucross(list(offspiringN[off]), offspiringN[chrRan]))
                    offspiring[str(ind)] = tmp2
                    indx2.append(str(ind))
                    tmp2 = []

            ###############################
#offsring dataset cross the new popoylation and indx2 is the index


    # evalouation each chromosome
    # we have to bult a graph for each cromosome and find the conected comunites
    # then we run modularity and storees it
    modularityy=[]
    result = pd.DataFrame()
    gp = []
    gpin=[]
    for i in indx2:  # we have 2 extra coloum 0 coloumn and node coloumn
        #chro = 'chromosom' + str(i)
        tmpgr=(nx.from_pandas_edgelist(offspiring, i, 'node'))
        concomponet=list(list(nx.connected_components(tmpgr)))
        modularityy.append(modularity(tmpgr,concomponet))
        gpin.append(i)
        #print(chro)
    gpdf['modularity']=modularityy
    #gpdf['concomp']=concomponet
    gpdf['indx'] =gpin
    # for i in range (0,len(gpdf['graph'])):
    #     # gr=gp[i]
    #     concomponet.append(modularity(gp[i], list(list(nx.connected_components(gp[i])))))
    #     ingp.append(i)
   # result['modularity'] = concomponet
    #result['indx']=ingp

    # take finala result the result for next generation
    result = pd.DataFrame()
    result = gpdf[gpdf['modularity'] > 0.5]
    b1=result.sort_values(by=['modularity'], ascending=False).head(1)
    bb=result.sort_values(by=['modularity'], ascending=False).head(4)

    indx = list(result['indx']) #for indexing the oofsring only from the result chosen creteria

    # we select the second generation all the chromosomu with modularity > 0.79

    # we select the second generation all the chromosomu with modularity > 0.79
    offspiringN = pd.DataFrame()
    bbf['node'] = cc['node']
    offspiringN=bbf  #next genertion offspingN with bbg chromosums

    #offspiringN['node'] = cc['node']
   #bbf= pd.DataFrame()  #clear the bbf
  #3  for b in bb['indx']: #next and next next generatrion bbf selected chromosom in idx
    #    bbf[b]=offspiring[b]


    #offspiringN.append(offspiringNf)

    for i in indx:    #chose the chromosom two the next generatin offsprinN
    #indx = resultF.index.values.tolist()
    # for i in resultF.index.values.tolist():
        offspiringN[i] = offspiring[(i)]
    print('Number of interation :',generation,generation,generation,generation,generation)
    offspiring=pd.DataFrame() #delete offspiring
    offspiring['node'] = cc['node'] #appen the nodes

    #mutation
    #fint he curent popoulation
    curpup=int(offspiringN.shape[1]*mutationRate)

    nummutation=int(mutationRate*offspiringN.shape[1])
    rinxx= indx[:]  #copy list
    muta=[]
    if len(indx)>0:
        while curpup>=0:
            random.shuffle(rinxx) #take random %persent times ffrom ofspring N mutatioon
            try:
                muta.append(rinxx.pop())
            except:
                 print("Mutation perfomed")
            curpup=curpup-1
        for i in muta:   #implemet the mutation
            offspiringN[i] = mutation(offspiringN[i], nodesA)  #replece a random gen  that is goint to cross to the next generation
        curpup=curpup-1

