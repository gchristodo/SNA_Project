import numpy as np
import pandas as pd
import networkx as nx
df = pd.read_csv('movies.csv')
#delete null values
df=df[df['director_name'].notnull()]

#Append to dataframe evaloution from gross of movie
evalo=[]
for row in df['gross']:
    if (row <100000) :evalo.append(1)
    elif (row >100000 and row <400000) :evalo.append(2)
    elif (row >400000 and row <1000000) :evalo.append(3)
    elif (row >1000000 and row <10000000) :evalo.append(4)
    elif (row >10000000 and row <50000000) :evalo.append(5)
    elif (row >50000000 and row <100000000) :evalo.append(6)
    elif (row >100000000 and row <200000000) :evalo.append(7)
    elif (row >200000000 and row <400000000) :evalo.append(8)
    elif (row >400000000):evalo.append(9)
    else:evalo.append(0)
df["eval"]=evalo

#Take the actor1 af dataframe1 and dataframe2 actor2,Alseo 4


df1 = df[['director_name','actor_1_name','gross','eval']]
df2=df[['director_name','actor_2_name','gross','eval']]

#rename coloumn
df2.columns = ['director_name', 'actor_1_name','gross','eval']

#merge df2 and df2


dff= df1.append(df2)
df3 = df[['director_name','actor_3_name','gross','eval']]


#rename coloumn
df3.columns = ['director_name', 'actor_1_name','gross','eval']
fdf= dff.append(df3)

fdf.columns = ['director_name', 'actor_name','gross,','eval']

graphData = fdf[['director_name','actor_name','eval']]


#Below we sum the actor each coloboration
fdfcnta=fdf.groupby('director_name')['actor_name'].count()
fdfcnta=['director_name', 'numA']

#reseting index
#fdf = fdf.reset_index()

#we can export 
# dff.to_csv('formatted-data.csv', date_format='%B %d, %Y')
fdf.to_csv('Direc_actor_data.csv',index=False)

#create the graph
g = nx.from_pandas_edgelist(fdf, 'director_name', 'actor_name')
G = nx.from_pandas_edgelist(graphData, 'director_name', 'actor_name', ['eval'])

#take the edges list weigts
list(G.edges(data=True))
            
            #returns the noubors list
def neib(ne):
    n=[]
    for i in ne:
        n.append(i)
    return n


# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    if clustering==[]:
        clusters=[]
    else:
        clusters = max(clustering) + 1
    modularity = 0 # Initialize total modularity value

    #Iterate over all clusters
    for i in range(clusters):

        # Get the nodes that belong to the i-th cluster (increase node id by 1)
        nodeList = [n+1 for n,v in enumerate(clustering) if v == i]

        # Create the subgraphs that correspond to each cluster				
        subG = G.subgraph(nodeList)
        temp1 = nx.number_of_edges(subG) / float(nx.number_of_edges(G))
        temp2 = pow(sum(nx.degree(G, nodeList).values()) / float(2 * nx.number_of_edges(G)), 2)
        modularity = modularity + (temp1 - temp2)
    return modularity

#example
node='Robert De Niro'

nei=g[node]

#take the ccoeficient neibhors the larger coefficeint/ return the node and the coefficent number

# cc for all neibors
# cc for all neibors
def clst(G, nodes):
    k = 0
    j = []
    n = []
    for nod in nodes:
        # print(nx.clustering(G,nod))
        # print ('node:'+str(nod) + '-clus_coeff:'+str(nx.clustering(G,nod)))
        # if k<float(nx.clustering(g,nod)):
        k = (nx.clustering(G, nod))
        # print('yeeeesfsfdsfdsfdsffdsfdsfdsfdgfdgfdgfddgfdgdfgsfdsfsfsdfsfsdfdsfsdfdsfee')
        # print ('node:'+str(nod) + '-clus_coeff:'+str(nx.clustering(g,nod)))
        # j=nod
        j.append(float(k))
        n.append(nod)

    return n, j



#take the maximum cc forthe neibor

#find the maximum cc
def nodech(nod):
    m=max(nod[1])
    inde=nod[1].index(max(nod[1]))
    no=nod[0][inde]

# the maximim cc for g,node is a[inde][0], and te chosen cluster is
    return no,inde

def nebigorList(G,no):
    allne=G.neighbors(no)
    return neib(allne)


#cc for all graph return for each node of the graf the node and the cc node with value cc

ccg=[]
node=[]
for i in g:
    nn=nebigorList(g,i)
    pp=clst(g,nn)
    mm=nodech(pp)
    node.append(i)
    ccg.append(mm)

df4=pd.DataFrame()
df4["node"]=node
df4["cc"]=ccg
print(df4)
grpahcc=pd.DataFrame()


node2=[]
ToNode=[]
cc2=[]
for i in range(0,len(df4)):
    node2.append(df4["node"][i])
    ToNode.append(df4["cc"][i][0])
    cc2.append(df4["cc"][i][1])
grpahcc["node"]=node2
grpahcc["Tonode"]=ToNode
grpahcc["cc"]=cc2


#graph cc is the seconfd graph for cluster coeficent now wwe can export the data to analyze to  gephi as edge list
grpahcc.to_csv('cc.csv',index=False)

g2 = nx.from_pandas_edgelist(grpahcc, 'node', 'Tonode')

comunitiesnode=[ c for c in sorted(nx.connected_components(g2), key=len, reverse=True)]
numcommunities=len(comunitiesnode)
     
#nx.draw(g)
node='Steven Spielberg'
#number of nodes 
pupula=len(g)

#we can ran for all nodes the Coeffieecnt to for each node to pick the node with the larger cc




#crsRate=0.01
#mutatuRate=0.01



    
