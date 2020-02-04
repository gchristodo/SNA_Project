import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib import cm,colors

from collections import defaultdict


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
        a = np.mean([ nx.shortest_path_length(G,source=node,target=n) 
                      for n in communities_t[communities[node]] ])
        
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
        return np.round(sil_coef, 3), np.round(max(sil_coef), 3)
    else:
        return sil_coef,1

def draw_communities(G, y_actual, title):
    """
    Function responsible to draw the nodes to a plot with assigned colors for each individual cluster
    Inputs
    ----------
    G : networkx graph
    y_actual : list with the ground truth
    title : The title of the plot
    """ 
    with open('karate_pos.json', 'r') as read_file:
        pos_ = json.loads(read_file.read())
    p_1 = list(map(int, pos_.keys()))
    p_2 = list(map(np.asarray, pos_.values()))
   
    pos = dict(zip(p_1, p_2))
    
    map_ = {val:i for i, val in enumerate(set(y_actual))}
    
    y_actual = list(map(lambda x: map_[x], y_actual))
    
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