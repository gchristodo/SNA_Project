# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:09:40 2020

@author: Christodoulou
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd             
import networkx as nx
from itertools import product
from itertools import combinations
import time

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
            'edges_change': True
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
            'modularity': 'GN', # GN / other / density
            'edges_change': True # True/False: m = 2*number_of_edges
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
            'modularity': 'GN', # GN / other / density
            'edges_change': True # True/False: m = 2*number_of_edges
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
            'modularity': 'GN', # GN / other / density
            'edges_change': True # True/False: m = 2*number_of_edges
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
            'modularity': 'GN', # GN / other / density
            'edges_change': True # True/False: m = 2*number_of_edges
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
        if not self.Graph:
            message = 'Graph not loaded'
        else:
            message = nx.info(self.Graph)
        return print(message)
    
    def remove_edges(self, ebc, graph, multiple_edges_removal=True):
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
            
    def get_density(self):
        return nx.density(self.Graph)
    
    def run_algorith(self):
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
            
                
## Creating a GN object
G = GirvanNewman(settings_jazz)

start_time = time.time()
## Running the algorithm
best_components, best_modularity = G.run_algorith()
print("--- %s seconds ---" % (time.time() - start_time))

