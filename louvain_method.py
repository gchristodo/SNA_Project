import networkx as nx
import matplotlib.pyplot as plt
from community import modularity
from datetime import datetime
import numpy as np



def louvain(G, visualize=False, tol=0.01):
   '''
   Implementation of the Louvain method that finds communities using graph modularity maximization.
   Arguments: 
      G: graph to apply the method
      visualize: boolean argument for intermediate graph visualization
      tol: tolerance threshold
   '''
   # Initialization
   start = datetime.now() 
   # Get neighbors
   neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}
   # Set initial communities
   communities = { node: node for node in neighbors.keys() }
   # Get nodes
   nodes = list(G.nodes())
   # Calculate initial modularity
   mod = modularity(communities, G)

   while True:
      for node in nodes:
         # List hosting the temporary modularity results between each node and its neighbor
         mod_list = []

         # Calculate modularities
         for n in neighbors[node]:
               tmp_communities = communities.copy()
               tmp_communities[node] = tmp_communities[n]
               tmp_mod = modularity(tmp_communities, G)
               mod_list.append(tmp_mod)

         # Get the maximum modularity of the temporary modularity calculations
         max_mod_ix = np.argmax(mod_list)
         tmp_mod = np.max(mod_list)

         # Get the neighbor offering the best modularity
         best_neighbor = neighbors[node][max_mod_ix]
         # In case the calculated modularity is greater than the initial
         # The node is absorbed by the neighbor along with all the nodes
         # that belonged in its cluster
         if tmp_mod > mod:
            communities[node] = communities[best_neighbor]
            for n, c in communities.items():
               if c == node:
                  communities[n] = communities[c]

      # Create the new graph updated by the new relationships
      G = nx.Graph()
      new_nodes = list(set([c for n, c in communities.items()]))
      G.add_nodes_from(new_nodes)
      for ex_node in communities.keys():
         for n in neighbors[ex_node]:
               # Draw an edge between the community and the community of the neighbor of the ex_node
               G.add_edge(communities[ex_node], communities[n])
     
      # Calculate new modularity
      new_mod = modularity(communities, G)

      if visualize:
         # Draw new graph
         communities_no = len(set([v for k, v in communities.items()]))
         nx.draw_spring(G,
                        cmap=plt.get_cmap('tab20'),
                        node_color=new_nodes,
                        with_labels=True,
                        title='New nodes: {},   modularity: {}, number of communities: {}'.format(new_nodes, new_mod, communities_no))
         plt.show()

      # Repeat until condition
      if new_mod < mod + tol:
         break
      else:
         mod = new_mod
   print('Finished community detection in {}'.format(datetime.now()-start))
   return communities