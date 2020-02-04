from sklearn import cluster
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm,colors
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import numpy as np

def draw_communities(G, y_actual, pos, title):
    """
    Function responsible to draw the nodes to a plot with assigned colors for each individual cluster
    Inputs
    ----------
    G : networkx graph
    y_actual : list with the ground truth
    pos : positioning as a networkx spring layout
        E.g. nx.spring_layout(G)
    title : The title of the plot
    """ 
    fig, ax = plt.subplots(figsize=(16,9))
    
    # Convert y_actual list to a dict where key=cluster, value=list of nodes in the cluster
    key = defaultdict(list)
    for node, value in enumerate(y_actual):
        key[value].append(node)
    
    # Normalize number of clusters for choosing a color
    norm = colors.Normalize(vmin=0, vmax=len(key.keys()))
    
    for cluster, members in key.items():
        nx.draw_networkx_nodes(G, pos,
                               nodelist=members,
                               node_color=cm.jet(norm(cluster)),
                               node_size=500,
                               alpha=0.8,
                               ax=ax)

    # Draw edges (social connections) and show final plot
    plt.title(title)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax,edge_labels=labels)

def graph_to_weighted(G):
    """
    Convert a networkx graph into a weighted matrix.
    Inputs: 
    ----------
    G : networkx graph
    """
    # Initialize weighted matrix with zeros
    weighted_mat = np.zeros((len(G), len(G)), dtype=int)
    
    for node in G:
        for neighbor in G.neighbors(node):
            try:
              weighted_mat[node][neighbor] = G[node][neighbor]['weight']
            except KeyError:
              weighted_mat[node][neighbor] = G[neighbor][node]['weight']
        weighted_mat[node][node] = 0

    return weighted_mat

def graph_to_degree_matrix(G):
    """
    Convert a networkx graph into an degree matrix.
    Inputs: 
    ----------
    G : networkx graph
    """
    # Initialize degree matrix with zeros
    degree_mat = np.zeros((len(G), len(G)), dtype=int)
    
    for node in G:
        for neighbor in G.neighbors(node):
            degree_mat[node][neighbor] = 0
        degree_mat[node][node] = G.degree[node]

    return degree_mat

def graph_to_adj_matrix(G):
    """
    Convert a networkx graph into an adj matrix.
    Inputs: 
    ----------
    G : networkx graph
    """
    # Initialize adj matrix with zeros
    adj_mat = np.zeros((len(G), len(G)), dtype=int)
    
    for node in G:
        for neighbor in G.neighbors(node):
            adj_mat[node][neighbor] = 1
        adj_mat[node][node] = 1

    return adj_mat

def draw_true_vs_pred(G, y_true, y_pred, pos, algo_name, ax):
    

    for val in range(len(y_true)):
        if y_pred is not None:
            if y_true[val] == y_pred[val]:
                node_color = [0, 1, 0]
                node_shape = 'o'
            else:
                node_color = [0, 0, 0]
                node_shape = 'X'
       
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[val],
                               node_color=node_color,
                               node_size=250,
                               alpha=0.7,
                               ax=ax,
                               node_shape=node_shape)
    
    # Draw edges and show final plot
    ax.set_title(algo_name)
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

def silhouette_score(communities, G):
    clusters = list(set(communities.values()))
    nodes = list(communities.keys())
    # Create the transposed communities dict
    communities_t = {c:[] for c in clusters}
    for node in nodes:
        communities_t[communities[node]].append(node)
    sil_coef = []
    for node in nodes:
        # calculate average inner distance: a(u)
        paths = []
        for n in communities_t[communities[node]]:
            try:
                paths.append(nx.shortest_path_length(G,source=node,target=n))
            except nx.NetworkXNoPath:
                pass
        a = np.mean([paths])
        # calculate minimum average outer distance: b(u)
        mean_outer_distances = []
        for c in clusters:            
            for n in communities_t[c]:   
                outer_distance = []
                if communities[node] != communities[n]:
                    outer_distance = []
                    # in case there is no path from node to n
                    try:
                        outer_distance.append(nx.shortest_path_length(G,source=node,target=n))
                    except nx.NetworkXNoPath:
                        pass
            if outer_distance:        
                mean_outer_distances.append(np.mean(outer_distance))
        if mean_outer_distances:
            b = np.min(mean_outer_distances)
            # calculate silhouette coefficient
            sil_coef.append( (b-a)/max(a,b) )
    # In case the dataset is homogenized
    if sil_coef:
        return np.round(max(sil_coef), 3)
    else:
        return 1