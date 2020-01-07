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
dff.columns = ['director_name', 'actor_name']

#we can export maybe the seconde
# dff.to_csv('formatted-data.csv', date_format='%B %d, %Y')
 dff.to_csv(('formatted2-data.csv',index=False)




#παραπανω κανουμε εξαγωγη σε csv και το ανοιγουμε με gephi για να δουμε τα cluster poy dhmhoyurgoyntai





#create the graph
g = nx.from_pandas_edgelist(dff, 'director_name', 'actor_name')
            
            #returns the noubors list
def neib(ne):
    n=[]
    for i in ne:
        n.append(i)
    return n

#example
node='Robert De Niro'

nei=g[node]

#take the ccoeficient neibhors the larger coefficeint/ return the node and the coefficent number
j=[]
 def clst(g,nodes):
    k=0
    j=[]
    n=[]
    for nod in g[nodes]:
        print(nx.clustering(g,nod))
        print ('node:'+str(nod) + '-clus_coeff:'+str(nx.clustering(g,nod)))
        #if k<float(nx.clustering(g,nod)):
        k=(nx.clustering(g,nod))
        #print('yeeeesfsfdsfdsfdsffdsfdsfdsfdgfdgfdgfddgfdgdfgsfdsfsfsdfsfsdfdsfsdfdsfee')
        #print ('node:'+str(nod) + '-clus_coeff:'+str(nx.clustering(g,nod)))
        #j=nod
        j.append(float(k))
        n.append(nod)
    return j,n


#find the maximum cc cluster coeffficent
def nodech(gra,nod):
    a=clst(gra,nod)
    m=max(a[0])
    inde=a[0].index(max(a[0]))

# the maximim cc for g,node is a[inde][0], and te chosen cluster is
    return a[1][inde],a[0][inde]
     
#nx.draw(g)
node2='Steven Spielberg'
#number of nodes 
pupula=len(g)

#we can ran for all nodes the Coeffieecnt to for each node to pick the node with the larger cc




crsRate=0.01
mutatuRate=0.01
