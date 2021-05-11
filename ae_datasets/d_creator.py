import networkx as nx
import numpy as np
import torch

import graph as gl
import random

OFFSET_4 = torch.from_numpy(np.array([[-5, 5], [5, 5], [-5, -5], [5, -5]]).astype(np.float32))
OFFSET_9 = torch.from_numpy(np.array([[-10, -10], [0, -10], [10, -10], [-10, 0], [0, 0],
                                      [10, 0], [-10, 10], [0, 10], [10, 10]]).astype(np.float32))
OFFSET_16 = torch.from_numpy(np.array([[(i % 4)*10 - 15, (i // 4)*10 - 15] for i in range(16)]).astype(np.float32))
OFFEST_DICT = {'mogg_4': OFFSET_4, 'mogg_9': OFFSET_9, 'mogg_16': OFFSET_16}


class Dataset:
    def __init__(self):
        self.graphs = None

    def create(self):
        pass

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


class DatasetErdosRenyi(Dataset):
    """
    We sample graphs of a given size N were all nodes and edges have a continuous value

    """
    def __init__(self, n_samples=None, n_nodes=10, n_edges=20, partition='train', directed=True):
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.partition = partition

        self.directed = directed

        if n_samples is None:
            if self.partition == 'train':
                self.seed = 0
                self.n_samples = 5000
            elif self.partition == 'val':
                self.seed = 1
                self.n_samples = 500
            elif self.partition == 'test':
                self.seed = 2
                self.n_samples = 500
            else:
                raise Exception("Wrong partition")
        else:
            self.n_samples = n_samples
            self.seed = 3
        self.graphs = self.create()

    def create(self):
        graphs = []
        random.seed(self.seed)
        for i in range(self.n_samples):
            G = nx.gnm_random_graph(self.n_nodes, self.n_edges, directed=False)
            if self.directed:
                G = G.to_directed()
            G = gl.networkx2graph(G)
            graphs.append(G)
        return graphs

class DatasetErdosRenyiNodes(Dataset):
    def __init__(self, n_samples=None, p=0.25, partition='train', overfit=False, directed=True):
        self.n_nodes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.n_samples = n_samples
        self.directed = directed
        self.p = p
        self.partition = partition
        if self.partition == 'train':
            self.seed = 0
            self.n_samples = 5000
        elif self.partition == 'val':
            self.seed = 1
            self.n_samples = 500
        elif self.partition == 'test':
            self.seed = 2
            self.n_samples = 500

        self.n_samplesxnodes = int(self.n_samples / len(self.n_nodes))
        if overfit:
            self.n_samplesxnodes = 10
            self.seed = 3


        self.graphs = self.create()

    def create(self):
        graphs = []
        random.seed(self.seed)
        for i in range(self.n_samplesxnodes):
            for n_nodes in self.n_nodes:
                G = nx.gnp_random_graph(n_nodes, self.p, directed=False)
                if self.directed:
                    G = G.to_directed()
                G = gl.networkx2graph(G)
                graphs.append(G)
        random.shuffle(graphs)
        return graphs


class GraphToArray(Dataset):
    """
    This class wraps a Dataset class and casts the graphs to array format
    """
    def __init__(self, dataset, undirected_graph=False):
        self.dataset = dataset
        self.undirected_graph = undirected_graph

    def sample(self, n_samples):
        graphs = self.dataset.sample(n_samples)
        arrays = []
        for graph in graphs:
            arrays.append(self.graph2array(graph))
        return torch.cat(arrays, dim=0)

    def graph2array(self, graph):
        '''Note: This function only works for graphs with the same number of nodes and edges, it has to be adapted to any type of graph'''
        nodes = torch.flatten(graph.nodes)
        if self.undirected_graph:
            edges = torch.flatten(graph.edge_attr[:, 0:graph.edge_attr.size(1)//2])
        else:
            edges = torch.flatten(graph.edge_attr)
        array = torch.cat([nodes, edges], dim=0)
        return array.unsqueeze(0)


class GraphBatchToGraph(Dataset):
    """
    This class wraps a Dataset class and casts the batch of graphs to a big one
    """
    def __init__(self, dataset, undirected_graph=False):
        self.dataset = dataset
        self.undirected_graph = undirected_graph

    def sample(self, n_samples):
        assert(n_samples == 1)
        graphs = self.dataset.sample(n_samples)
        return graphs[0]


class DatasetCommunity(Dataset):
    def __init__(self, n_samples=None, partition='train', num_communities=2):

        self.partition = partition
        self.num_communities = num_communities

        if n_samples is None:
            if partition == 'train':
                self.n_samples = 5000
                seed = 0
            elif partition == 'val':
                self.n_samples = 500
                seed = 1
            elif partition == 'test':
                self.n_samples = 500
                seed = 2
            else:
                raise Exception("Wrong seed")
        else:
            self.n_samples = n_samples
            seed = 3
        self.graphs = self.create(seed)


    def create(self, seed=None):
        np.random.seed(seed)
        graphs = []
        print('Creating dataset with ', self.num_communities, ' communities')

        # c_sizes = [15] * num_communities
        for k in range(self.n_samples):
            c_sizes = np.random.choice([6, 7, 8, 9, 10], self.num_communities)
            graphs.append(n_community(c_sizes, p_inter=0.01))
        return graphs



def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_component_subgraphs(G))
    #communities = list(G.subgraph(c) for c in nx.connected_components(G))[0]
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    #print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    G = G.to_directed()
    G = gl.networkx2graph(G)
    return G




def max_n_nodes(graphs):
    max_nodes = 0
    for graph in graphs:
        n_nodes = len(graph.nodes)
        if n_nodes > max_nodes:
            max_nodes = n_nodes
    return max_nodes



if __name__ == "__main__":
    gnf_dataset_keys = ['ego_small', 'community_small', 'graph_rnn_protein', 'ego', 'grid', 'community', 'community_medium']

    dataset = DatasetErdosRenyiNodes(partition='test', overfit=True)
    for graph in dataset.graphs:
        gl.plot_graph(graph)


