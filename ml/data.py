import torch
import dgl
import dgl.data
import urllib.request
import pandas as pd
from dgl.data import DGLDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

""" urllib.request.urlretrieve(
    'https://data.dgl.ai/tutorial/dataset/graph_edges.csv', './graph_edges.csv')
urllib.request.urlretrieve(
    'https://data.dgl.ai/tutorial/dataset/graph_properties.csv', './graph_properties.csv')
edges = pd.read_csv('./graph_edges.csv')
properties = pd.read_csv('./graph_properties.csv')

edges.head()

properties.head() """

class SyntheticDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
        edges = pd.read_csv('./edge.csv')
        properties = pd.read_csv('./graph.csv')
        attributes = pd.read_csv('./node.csv')
        # global_attributes = pd.read_csv('./global_attributes.csv')
        self.graphs = []
        self.labels = []

        # global_attributes = global_attributes['global_att'].to_numpy()
        # print(global_attributes)

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')
        attributes_group = attributes.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]
            #print(label)
        
            attributes_of_id = attributes_group.get_group(graph_id)
            type = torch.from_numpy(attributes_of_id['type_id'].to_numpy())       
            type = F.one_hot(type, num_classes=19)
            type = torch.tensor(type, dtype = torch.float)
            prop = torch.from_numpy(attributes_of_id['prop'].to_numpy())
            prop = torch.tensor(prop, dtype = torch.float)
            #link = torch.from_numpy(edges_of_id['etype'].to_numpy())
            #link = torch.tensor(link, dtype = torch.float)

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes_dict[graph_id])
            g = dgl.to_bidirected(g)
            #g = dgl.add_self_loop(g)
            g.ndata['type'] = type
            g.ndata['prop'] = prop
            #g.edata['link'] = link
            # g = dgl.to_bidirected(g)
            # G = dgl.to_networkx(bg)
            # plt.figure(figsize=[15,7])
            # nx.draw(G)
            self.graphs.append(g)
            self.labels.append(label)


        # Convert the label list to tensor for saving.
        self.labels = torch.FloatTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

dataset = SyntheticDataset()
print(dataset)
graph, label = dataset[0]
print(graph)
G = dgl.to_networkx(graph)
print(G.nodes)
print(np.argmax(graph.ndata['type'],axis=1).tolist())
print(label)
