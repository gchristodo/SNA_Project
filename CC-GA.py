import numpy as np
import pandas as pd
import networkx as nx
df = pd.read_csv('movies.csv')
#delete null values
df=df[df['director_name'].notnull()]

#Take the actor1 af dataframe1 and dataframe2 actor2,Alseo 4


df1 = df[['director_name','actor_1_name']]
df2=df[['director_name','actor_2_name']]

#rename coloumn
df2.columns = ['director_name', 'actor_1_name']

#merge df2 and df2


dff= df1.append(df2)
df3 = df[['director_name','actor_3_name']]
#rename coloumn
df3.columns = ['director_name', 'actor_1_name']
fdf= dff.append(df3)

fdf.columns = ['director_name', 'actor_name']

#reseting index
#fdf = fdf.reset_index()

#we can export 
# dff.to_csv('formatted-data.csv', date_format='%B %d, %Y')
fdf.to_csv('Direc_actor_data.csv',index=False)

#create the graph
g = nx.from_pandas_edgelist(fdf, 'director_name', 'actor_name')
            
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

def clst(g,nodes):
    k=0
    j=[]
    n=[]
    for nod in nodes:
        #print(nx.clustering(g,nod))
        #print ('node:'+str(nod) + '-clus_coeff:'+str(nx.clustering(g,nod)))
        #if k<float(nx.clustering(g,nod)):
        k=(nx.clustering(g,nod))
        #print('yeeeesfsfdsfdsfdsffdsfdsfdsfdgfdgfdgfddgfdgdfgsfdsfsfsdfsfsdfdsfsdfdsfee')
        #print ('node:'+str(nod) + '-clus_coeff:'+str(nx.clustering(g,nod)))
        #j=nod
        j.append(float(k))
        n.append(nod)
    return j,n


#find the maximum cc
def nodech(gra,nod):
    a=clst(gra,nod)
    m=max(a[0])
    inde=a[0].index(max(a[0]))

# the maximim cc for g,node is a[inde][0], and te chosen cluster is
    return a[1][inde],a[0][inde]
     
#nx.draw(g)
node='Steven Spielberg'
#number of nodes 
pupula=len(g)

#we can ran for all nodes the Coeffieecnt to for each node to pick the node with the larger cc




#crsRate=0.01
#mutatuRate=0.01



    
