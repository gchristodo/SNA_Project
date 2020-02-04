from sklearn import cluster
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import scipy
from numpy.linalg import inv
from numpy import linalg as LA
from utility_functions import *
from community import modularity


class SpectralClustering:
    """# Custom Implemented Spectral Clustering Algorithms

    **In order to familiarize with Spectral Clustering we have implemented the original version by Calculating the Laplacian matrix and the eigenvalues / eigenvectors**

    """

    def __init__(self, cluster, startingM):
       self.cluster = cluster
       self.startingM = startingM
    
    def laplacian(self,A):
       """
       Computes the symetric normalized laplacian.
       L = D^{-1/2} A D{-1/2}
       Inputs: 
       ----------
       A : Affinity or Adj matrix
       """
       D = np.zeros(A.shape)
       w = np.sum(A, axis=0)
       D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
       return D.dot(A).dot(D)

    def k_means(self,X, n_clusters):
       """
       Computes KMeans clustering algorithm.
       Inputs: 
       ----------
       X : array-like or sparse matrix, shape=(n_samples, n_features)
       n_clusters : The number of clusters
       """
       kmeans = KMeans(n_clusters=n_clusters, random_state=1231)
       return kmeans.fit(X).labels_

    def spectral_clustering(self,affinity, n_clusters):
       """
       Calculates Spectral Clustering
       Inputs: 
       ----------
       affinity : Affinity or Adj matrix
       n_clusters : The number of clusters
       """
       L = self.laplacian(affinity)
       eig_val, eig_vect = scipy.sparse.linalg.eigs(L, n_clusters)
       X = eig_vect.real
       rows_norm = np.linalg.norm(X, axis=1, ord=2)
       Y = (X.T / rows_norm).T
       labels = self.k_means(Y, n_clusters)
       return labels

    def RAM(self,M, e, startingIndexes, printMiddleGaps):
       """# RAM Spectral Clustering Algorithm via Simulated Mixing
       (https://ieeexplore.ieee.org/document/8331142)
    
       The current implementation is not full percent aligned with the pseudo-code written in the published article.
    
       At the implementation we have faced many difficulties to understand the way the algorithm will work and how the results will be produced and returned from the function.
    
       Specifically although the algorithm indicates on 2 Input variables:
    
       *   M: The Similarity Matrix calculates as: (1-a)I + a * inv(D) * W
           where I is the Identity matrix, a is a custom variable with no further          input on its values range, D is the diagonal degree Matrix and W is the         weighted Matrix
       *   e0: The initial tolerance
    
       The Idea of the algorithm is to use Linear Algebra transformation with random created vectors in order to "mix" the nearest points and create clusters. The algorithm does not take as input the number of clusters k
       """
       # We are using 2 global variables in order to get the outcome of the algorithm
       # cluster will take all the final produced clusters
       # startingM will be used to reinitilize the recursive procedure
       global cluster
       global startingM
       n = M.shape[0]
       # b is another dynamic values that determines the way the cluster behaves
       # as it is used in order to decrease the tolerance we have defined. If it is 
       # the tolerance will quickly reach a state where it will terminate the execution
       b = 100
       x = np.random.uniform(0,b,n)
       e0 = e
       emin = 0.000001
       tmax = 600
       t = 0
       np.seterr(all='raise')
       while True:
          while True:
             if t == 0:
                xt = M.dot(x)
                yt1 = LA.norm(xt - x)
                t += 1
                continue
             # We create a vector simulating an eigenvector from the x random generated vector 
             # which we have calculated the dot product with the similarity Matrix
             x = xt
             yt = yt1
             xt = M.dot(x)
             yt1 = LA.norm(xt - x)
             t += 1
             try:
                difference = abs(yt1 - yt)
             except FloatingPointError:
                break
             if difference <= e0:
                break
          # Calculation of gap. If a positive gap is found it will be used to bipartite a graph
          gap = []
          sortedXt = np.sort(xt)
          for i in range(len(xt) -1):
             arg1 = sortedXt[i+1] - sortedXt[i]
             arg2 = b / (2 * n)
             if arg1 >= arg2:
                gap.append(arg1)
             else:
                gap.append(0)
          try:
             maxGap = np.amax(gap)
          except ValueError:  #raised if `gap` is empty or very small
             maxGap = 0
          if e0 <= emin or t >= tmax:
             return
          e0 = e0 / 2
          if maxGap > 0:
             break
       Mi1 = []
       Mi2 = []
       ind1 = []
       ind2 = []
       flag = 1
       # At this stage the algorithm has found a gap
       # After that we need to bipartite the graph based on the maximum value of gap vector
       # Although in order not to lose the indexing of the points and return their label
       # we have to keep track of the indices
       for val in range (0, len(xt) -1):
          # The article does not mention the exact steps to bipartite the graph
          # So we are using the maxGap / 2 in order to split the graph
          if abs(xt[val] - xt[val+1]) > maxGap / 2 and flag == 1:
             Mi1.append(xt[val])
             if len(startingIndexes) == 0:
                ind1.append(val)
             else:
                ind1.append(startingIndexes[val])
             flag = 2
          elif abs(xt[val] - xt[val+1]) > maxGap / 2 and flag == 2:
             Mi2.append(xt[val])
             if len(startingIndexes) == 0:
                ind2.append(val)
             else:
                ind2.append(startingIndexes[val])
             flag = 1
          else:
             if flag == 1:
                Mi1.append(xt[val])
                if len(startingIndexes) == 0:
                   ind1.append(val)
                else:
                   ind1.append(startingIndexes[val])
             elif flag == 2:
                Mi2.append(xt[val])
                if len(startingIndexes) == 0:
                   ind2.append(val)
                else:
                   ind2.append(startingIndexes[val])
       if flag == 1:
          Mi1.append(xt[val+1])
          if len(startingIndexes) == 0:
             ind1.append(val+1)
          else:
             ind1.append(startingIndexes[val+1])
       elif flag == 2:
          Mi2.append(xt[val+1])
          if len(startingIndexes) == 0:
             ind2.append(val+1)
          else:
             ind2.append(startingIndexes[val+1])
       if printMiddleGaps == True:
           print("-----------------------------------------------------")
           print("First Graph")
           print(Mi1)
           print("-----------------------------------------------------")
           print("Second Graph")
           print(Mi2)
           print("-----------------------------------------------------")
           print("Maximum Gap")
           print(maxGap)
       # After we bipartite the graph we have to re-normalize the 2 splitted graphs
       # and continue the recursion. Though there aren't any steps for the re-normalization
       # of the graph. So in order to proceed we take a slice of the starting similarity matrix
       # based on the bipartite indexes we have calculated and start the recursion
       # The idea is that the random generated vectors will find further differences
       Mi1Final = np.zeros((len(Mi1),len(Mi1)))
       Mi2Final = np.zeros((len(Mi2),len(Mi2)))
       for i in range(len(Mi1)-1):
          for j in range(len(Mi1)-1):
             Mi1Final[i][j] = startingM[ind1[i]][ind1[j]]
       for i in range(len(Mi2)-1):
          for j in range(len(Mi2)-1):
             Mi2Final[i][j] = startingM[ind2[i]][ind2[j]]
       #row_sums = Mi1Final.sum(axis=1)
       #Mi1Final = Mi1Final / row_sums[:, np.newaxis]
       #row_sums = Mi2Final.sum(axis=1)
       #Mi2Final = Mi2Final / row_sums[:, np.newaxis]
       C = self.RAM(np.asarray(Mi1Final), e,ind1,printMiddleGaps)
       if C is None:
          self.cluster.append(ind1)
       C = self.RAM(np.asarray(Mi2Final), e,ind2,printMiddleGaps)
       if C is None:
          self.cluster.append(ind2)
       return self.cluster
    
    def printMetrics(self, algorithm, dataset, mod, sil):
       print("DATASET: %s " % dataset)
       print("")
       print("   %s Modularity: %.3f" % (algorithm, mod))
       print("   %s Silhouette: %.3f" % (algorithm, sil))
       print("")

# Main program
if __name__ == '__main__':
    """
       Loop through all the datasets and print the Silhouette and Modularity of the clustering
    """
    float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    
    datasets = ['contiguous-usa.csv','dolphins.csv','jazz.csv','karate.csv','ram_sample.csv','euroroad.csv']
    for dataset in datasets:
       if dataset == 'ram_sample.csv':
           RAM_sample_M = pd.read_csv("ram_sample.csv", header=None)
           RAM_sample_M = RAM_sample_M.to_numpy()
           startingM = RAM_sample_M
           cluster = []
           e = 0.1
           SC = SpectralClustering(cluster,startingM)
           RAM_sample_results = SC.RAM(RAM_sample_M,e,[],True)
           continue
       df = pd.read_csv(dataset)
       if dataset == 'karate.csv':
           G = nx.from_pandas_edgelist(df, 'target', 'source',['weight'])
           G = nx.convert_node_labels_to_integers(G)
           W = graph_to_weighted(G)
       else:
           G = nx.from_pandas_edgelist(df, 'target', 'source')
           G = nx.convert_node_labels_to_integers(G)
           W = graph_to_adj_matrix(G)
       cluster  = []
       D = graph_to_degree_matrix(G)
       n = D.shape[0]
       I = np.identity(n)
       a = 1
       e = 0.1
       M = (1-a) * I + a * inv(D).dot(W)
       startingM = M
       SC = SpectralClustering(cluster,startingM)
       C = None
       while C is None:
          C = SC.RAM(M,e,[],False)
          a -= 0.1
          M = (1-a) * I + a * inv(D).dot(W)
          startingM = M
          SC = SpectralClustering(cluster,startingM)
       communities = {}
       for i in range(0, len(C)):
          for j in range(0,len(C[i])):
             communities[C[i][j]] = i
       
       SC.printMetrics(dataset, "RAM", modularity(communities, G), silhouette_score(communities, G))
       
       k_clusters = 2
       adj_mat = graph_to_adj_matrix(G)
       custom_SC_results = SC.spectral_clustering(adj_mat, k_clusters)
       communities = {}
       for i in range(0, len(custom_SC_results)):
          communities[i] = custom_SC_results[i]
       SC.printMetrics(dataset, "Custom SC", modularity(communities, G), silhouette_score(communities, G))