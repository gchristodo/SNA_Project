# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:09:40 2020

@author: Christodoulou
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm,colors
import pandas as pd             
import networkx as nx
from itertools import product
from itertools import combinations
from collections import defaultdict
import time
import json
from matplotlib.axes._axes import _log as matplotlib_axes_logger

settings_karate = {
            'graph_file': {
                'name':'karate_club',
                'type':'embedded', # embedded/csv/txt
                },
            'graph_settings':{
                'delimiter': None,
                'create_using':None,
                'node_type':None,
                'edge_type':None,
                'encoding':'utf-8',
                'source': None,
                'target': None,
                'edge_attr': 'weight'
                },
            'multiple_edges_removal': True,
            'modularity': 'GN', # GN / other
            'edges_change': False
            }

settings_usa = {
            'graph_file': {
                'name':'contiguous-usa.txt',
                'type':'txt', # embedded/csv/txt
                },
            'graph_settings':{
                'delimiter': None,
                'create_using':nx.Graph(),
                'node_type':int,
                'edge_type':None,
                'encoding':'utf-8',
                'source': None,
                'target': None,
                'edge_attr': 'weight'
                },
            'multiple_edges_removal': True,
            'modularity': 'GN', # GN / other 
            'edges_change': False # True/False: m = 2*number_of_edges
            }

settings_jazz = {
            'graph_file': {
                'name':'jazz.txt',
                'type':'txt', # embedded/csv/txt
                },
            'graph_settings':{
                'delimiter': None,
                'create_using':nx.Graph(),
                'node_type':int,
                'edge_type':None,
                'encoding':'utf-8',
                'source': None,
                'target': None,
                'edge_attr': 'weight'
                },
            'multiple_edges_removal': True,
            'modularity': 'GN', # GN / other
            'edges_change': False # True/False: m = 2*number_of_edges
            }

settings_euroroad = {
            'graph_file': {
                'name':'euroroad.txt',
                'type':'txt', # embedded/csv/txt
                },
            'graph_settings':{
                'delimiter': None,
                'create_using':nx.Graph(),
                'node_type':int,
                'edge_type':None,
                'encoding':'utf-8',
                'source': None,
                'target': None,
                'edge_attr': 'weight'
                },
            'multiple_edges_removal': True,
            'modularity': 'GN', # GN / other 
            'edges_change': False # True/False: m = 2*number_of_edges
            }

settings_dolphins = {
            'graph_file': {
                'name':'dolphins.csv',
                'type':'csv', # embedded/csv/txt
                },
            'graph_settings':{
                'delimiter': None,
                'create_using':nx.Graph(),
                'node_type':None,
                'edge_type':None,
                'encoding':'utf-8',
                'source': 'source',
                'target': 'target',
                'edge_attr': None
                },
            'multiple_edges_removal': True,
            'modularity': 'GN', # GN / other
            'edges_change': False # True/False: m = 2*number_of_edges
            }



class GirvanNewman:
    def __init__(self, settings):
        self._settings = settings
        self.Graph_file = settings['graph_file']['name']
        self.Graph_type = settings['graph_file']['type']
        self.Graph = None
        self._before_quality = 0
        self._after_quality = 0
        self._num_of_edges = 0
        self._num_of_nodes = 0
        self._m = 0
        self._A = None
        self._in_degree = 0  # For GN modularity
        self._out_degree = 0 # For GN modularity
        self._degree = 0 # For other formula of modularity
        self._list_graph_nodes = None
        
    def create_graph(self):
        '''
        Method that creates a graph from embedded nx.graph or csv, txt file

        Raises
        ------
        ValueError
            If file is not in txt or csv format.

        Returns
        -------
        TYPE
            nx.Graph or nx.Digraph.

        '''
        if self._settings['graph_file']['type'] not in ['embedded', 'csv', 'txt']:
            raise ValueError('Input file should be only embedded or in csv or txt format')
        if self.Graph_file=='karate_club':
            self.Graph = nx.karate_club_graph()
        elif self.Graph_type=='txt':
            try:
                self.Graph = nx.read_edgelist(self.Graph_file, 
                                              delimiter=self._settings['graph_settings']['delimiter'], 
                                              create_using=self._settings['graph_settings']['create_using'], 
                                              nodetype=self._settings['graph_settings']['node_type'], 
                                              data=True, 
                                              edgetype=self._settings['graph_settings']['edge_type'], 
                                              encoding=self._settings['graph_settings']['encoding'])
            except:
                raise ValueError('Current type is not txt or settings are wrong.')
        else:
            try:
                df = pd.read_csv(self.Graph_file)
                self.Graph = nx.from_pandas_edgelist(df, 
                                                     source=self._settings['graph_settings']['source'], 
                                                     target=self._settings['graph_settings']['target'], 
                                                     edge_attr=self._settings['graph_settings']['edge_attr'], 
                                                     create_using=self._settings['graph_settings']['create_using'])
            except:
                raise ValueError('Current type is not csv or settings are wrong.')
        self._num_of_nodes = self.Graph.number_of_nodes()
        self._list_graph_nodes = list(self.Graph.nodes)
        # Calculating the initial m for modularity purposes. This doesn't change over iteration
        if type(self.Graph) == nx.Graph:
            self._m = 2.*(self.Graph.number_of_edges())
        elif type(self.Graph) == nx.DiGraph:
            self._m = 1.*(self.Graph.number_of_edges())
        else:
            raise ValueError("Graph type is not valid.")        
        return self.Graph
    
    def update_Graph(self, graph):
        '''
        Α method that updates adjacency matrix, inner degree, outer degree 
        and number of edges per modularity iteration

        Parameters
        ----------
        graph : nx.graph object

        Raises
        ------
        ValueError
            Graph type is not valid if graph is not directed or undirected.

        Returns
        -------
        A : TYPE
            Adjacency matrix of graph.
        in_degree : dictionary
            A dictionary: key node, value incoming edges.
        out_degree : dictionary
             A dictionary: key node, value outgoing edges.
        num_of_edges : float
            number of edges multiplied by 2 or 1 for modularity function .

        '''
        A = nx.adj_matrix(graph)
        if type(graph) == nx.Graph:
            in_degree = out_degree = dict(nx.degree(graph))
            num_of_edges = 2.*(graph.number_of_edges())
        elif type(graph) == nx.DiGraph:
            in_degree = dict(graph.in_degree())
            out_degree = dict(graph.out_degree())
            num_of_edges = 1.*(graph.number_of_edges())
        else:
            raise ValueError("Graph type is not valid.")
        return A, in_degree, out_degree, num_of_edges
            
    def info(self):
        '''
        A method that prints out the graph information

        Returns
        -------
        TYPE: String
            Information of the graph, like nx.info(G).

        '''
        if not self.Graph:
            message = 'Graph not loaded'
        else:
            message = nx.info(self.Graph)
        return print(message)
    
    def remove_edges(self, ebc, graph, multiple_edges_removal=True):
        '''
        Method that removes edges from the graph based on Edge Betweenness 
        Centrality

        Parameters
        ----------
        ebc : dictionary
            A dictionary with edge as key and their betweenness centrality 
            as value.
        graph : nx.graph object
            Our graph.
        multiple_edges_removal : boolean, optional
            DESCRIPTION. The default is True. Removes all edges with the
            highest EBC if True. Removes only one edge if set to False

        Raises
        ------
        ValueError
            Edge Betweeness Centrality should be a dictionary.
            Graph should be a nx.classes.graph.Graph object

        Returns
        -------
        None.

        '''
        if not isinstance(ebc, dict):
            raise ValueError('Edge Betweeness Centrality should be a dictionary.')
        if not isinstance(graph, nx.classes.graph.Graph):
            raise ValueError('Graph should be a nx.classes.graph.Graph object.')
        max_centrality_value = sorted(ebc.values())[-1]
        if multiple_edges_removal:
            for node, value in ebc.items():
                if value==max_centrality_value:
                    graph.remove_edge(node[0],node[1])
        else:
            for node, value in ebc.items():
                if value==max_centrality_value:
                    graph.remove_edge(node[0],node[1])
                    break
    
    def gn_iteration(self, multiple_edges_removal=True):
        '''
        Method that simulates the Girvan Newman algorithm

        Parameters
        ----------
        multiple_edges_removal : boolean, optional
            DESCRIPTION. The default is True. Removes all edges with the
            highest EBC if True. Removes only one edge if set to False

        Returns
        -------
        None.

        '''
        init_num_components = nx.number_connected_components(self.Graph)
        # print(nx.number_of_edges(self.Graph))
        print("Initial Components")
        print(init_num_components)
        print("------------------")
        num_components = init_num_components
        while num_components <= init_num_components:
            EBC = nx.edge_betweenness_centrality(self.Graph, weight=self._settings['graph_settings']['edge_attr']) # Edge Betweeness Centrality
            if len(EBC.values())==0:
                break
            self.remove_edges(EBC, self.Graph, multiple_edges_removal=multiple_edges_removal)
            num_components = nx.number_connected_components(self.Graph)

        
    def get_modularity(self, degree):
        '''
        A method for evaluating the quality of the clusters formed. (modularity)

        Parameters
        ----------
        degree : dictionary
            Degree of the graph, usable only in "other" modularity function.

        Returns
        -------
        flaot
            A number from -1/2 to 1

        '''
        self._A, self._in_degree, self._out_degree, self._num_of_edges = self.update_Graph(self.Graph)
        A = self._A
        print(A)
        new_degree = self._in_degree
        if self._num_of_edges==0:
            return 0
        total_Q = 0
        communities = list(nx.connected_components(self.Graph))
        if self._settings['modularity']=='GN':
            for community in communities:
                temp_comm = list(community)
                list_of_connected_nodes = list(combinations(temp_comm, 2))
                Q = 0
                for pair_of_nodes in list_of_connected_nodes:
                    node_index = (self._list_graph_nodes.index(pair_of_nodes[0]), self._list_graph_nodes.index(pair_of_nodes[1]))
                    alpha_i_j = self._A[node_index[0], node_index[1]]
                    fraction = (self._in_degree[pair_of_nodes[0]]*self._out_degree[pair_of_nodes[1]])/self._num_of_edges
                    Q += alpha_i_j - fraction
                total_Q += Q
            if self._settings['edges_change']:
                m = self._num_of_edges
            else:
                m = self._m
            modularity = total_Q/m
        elif self._settings['modularity']=='other':
            for community in communities:
                temp_comm = list(community)
                e = 0
                a = 0
                for node in temp_comm:
                    e += degree[node]
                    a += new_degree[node]
                total_Q += (e - ((a*a)/self._num_of_edges))
            if self._settings['edges_change']:
                m = self._num_of_edges
            else:
                m = self._m
            modularity = total_Q/m
        else:
            modularity = nx.density(self.Graph)
        return modularity
            
    
    def run_algorith(self):
        '''
        A method that runs the Girvan Newman algorithm with modularity evaluation

        Returns
        -------
        best_components : list
            A list of dictionaries that contain nodes of each community.
        before_quality : float
            The quality of the communities formed.

        '''
        self.create_graph()
        self._A, self._in_degree, self._out_degree, self._num_of_edges = self.update_Graph(self.Graph)
        original_degree = self._in_degree
        if self._settings['modularity']=='density':
            before_quality = nx.density(self.Graph)
        else:
            before_quality = self._before_quality
        after_quality = self.get_modularity(original_degree)
        while (self.Graph.number_of_edges()>0) and (after_quality >= before_quality) :
            before_quality = after_quality
            self.gn_iteration(self._settings['multiple_edges_removal'])
            after_quality = self.get_modularity(original_degree)
            print("After_Modularity: ", after_quality)
            print("Before_Modularity: ", before_quality)
            print("---------------------------")
            print("Components: ", list(nx.connected_components(self.Graph)))
            if after_quality > before_quality:
                best_components = list(nx.connected_components(self.Graph))
            
        return (best_components, before_quality)
            
    def communities_dic(self, components):
        '''
        A method that assigns each node to a community

        Parameters
        ----------
        components : list
            A list of dictionaries that contain nodes of each community.

        Raises
        ------
        ValueError
            Components should be a list.

        Returns
        -------
        community_dic : dictionary
            A dictionary: key->node, value->community in which the node belongs
            to.

        '''
        if not isinstance(components, list):
            raise ValueError("Components should be a list.")
        community_dic = {}
        for component in components:
            for node in component:
                community_dic[node] = components.index(component)
        return community_dic
    
    def silhouette_score(self, communities, G):
        '''
        A method that calculates the silhouette score

        '''
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
            return np.round(sil_coef, 3), np.round(max(sil_coef), 3)
        else:
            return sil_coef,1
        
    def draw_communities(self, G, y_actual, title):
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
        
## Creating a GN object
G = GirvanNewman(settings_jazz)

start_time = time.time()
original_graph = G.create_graph()
## Running the algorithm
best_components, best_modularity = G.run_algorith()
print("--- %s seconds ---" % (time.time() - start_time))
# Assigning each node to a community
my_com_dic = G.communities_dic(best_components)

values = [my_com_dic[node] for node in original_graph.nodes()]
## Drawing the graph with the formed communities
# Drawing works only with karate_club due to katate_pos.json
# If you want to draw other networks, make sure u have the equivalent json
# G.draw_communities(original_graph, values, "Girvan Newman: Zachary's Karate Club")

silhouette = G.silhouette_score(my_com_dic, original_graph)
print("Silhouette: ", silhouette)


