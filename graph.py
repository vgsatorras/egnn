import torch
import networkx as nx
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, nodes, edges='fc', edge_attr=None):
        '''

        :param nodes: tensor, size() --> (n_nodes, n_features)
        :param edges: [Long Tensor (rows), Long Tensor (cols)]
        :param edge_attr: tensor, size() --> (n_edges, n_features)
        '''

        self.nodes = nodes
        if edges == 'fc':
            self.edges = self._create_fc_edges()
        else:
            self.edges = edges
        self.edge_attr = edge_attr
        self._check_attr_validity()
        #self.adjacency = self._create_adjacency()

        self.adjacency, self.edges_dense, self.edge_attr_dense = None, None, None

    def set_edge_attr(self, edge_attr):
        self.edge_attr = edge_attr
        self.check_validity()

    def get_num_nodes(self):
        return self.nodes.size(0)

    def get_num_edges(self):
        return self.edges[0].size(0)

    def get_node_nf(self):
        return self.nodes.size(1)

    def get_edges_nf(self):
        if self.edge_attr is None:
            return 1
        else:
            return self.edge_attr.size(1)

    def get_total_size(self):
        return self.get_num_nodes()*self.get_node_nf() + self.get_num_edges()*self.get_edges_nf()

    def get_adjacency(self, loops=False):
        adjacency = self._create_adjacency(loops)
        return adjacency

    def get_dense_graph(self, store=True, loops=False):
        if self.adjacency is None or self.edges_dense is None or self.edge_attr_dense is None:
            adjacency = self.get_adjacency(loops)
            edges_dense, edge_attr_dense = self._dense2attributes(self.get_num_nodes(), adjacency)
            if store:
                self.edges_dense, self.edge_attr_dense, self.adjacency = edges_dense, edge_attr_dense, adjacency
            return self.nodes, edges_dense, edge_attr_dense, adjacency
        else:
            return self.nodes, self.edges_dense, self.edge_attr_dense, self.adjacency


    def _create_adjacency(self, loops):
        if self.edge_attr is not None:
            raise Exception('To be implemented')
        n_nodes = len(self.nodes)
        adjacency = sparse2dense(n_nodes, self.edges)
        if loops:
            adjacency = adjacency + torch.eye(n_nodes)
        else:
            adjacency = adjacency * (1 - torch.eye(n_nodes))
        return adjacency

    def _dense2attributes(self, n_nodes, adj_dense):
        edge_attr = torch.zeros((n_nodes ** 2 - n_nodes, 1))
        rows, cols = [], []
        counter = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    rows.append(i)
                    cols.append(j)
                    edge_attr[counter, 0] = adj_dense[i, j]
                    counter += 1
        edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
        return edges, edge_attr

    def _create_fc_edges(self):
        e_rows = []
        e_cols = []
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                e_rows.append(i)
                e_cols.append(j)
        e_rows += e_cols
        e_cols += e_rows[0:len(e_rows) // 2]
        return [torch.LongTensor(e_rows), torch.LongTensor(e_cols)]

    """
    def _flat_adjacency(self, adjacency):
        n_nodes = len(adjacency)
        flat_adj = torch.zeros((n_nodes ** 2 - n_nodes, 1))
        counter = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    flat_adj[counter] = adjacency[i, j]
                    counter += 1
        return flat_adj
    """

    def _check_attr_validity(self):
        if self.edge_attr is not None:
            assert len(self.edges[0]) == self.edge_attr.size(0)

def networkx2graph(G_nx):
    mapping = {}
    for idx, node in enumerate(G_nx.nodes):
        mapping[node] = idx
    G_nx = nx.relabel_nodes(G_nx, mapping)
    nodes = torch.ones((G_nx.number_of_nodes(), 1))
    rows = torch.LongTensor([edge[0] for edge in G_nx.edges])
    cols = torch.LongTensor([edge[1] for edge in G_nx.edges])
    edges = [rows, cols]
    graph = Graph(nodes, edges, edge_attr=None)
    return graph


def graph2networkx(G):
    edge_list = [(G.edges[0][i].item(), G.edges[1][i].item()) for i in range(len(G.edges[0]))]
    G_nx = nx.Graph(edge_list)
    return G_nx


def plot_graph(graph):
    graph_nx = graph2networkx(graph)
    nx.draw(graph_nx)
    plt.show()

def plot_networkx(graph_nx, path='graph_plot.png'):
    nx.draw(graph_nx)
    plt.savefig(path)
    plt.clf()

#################################
### Adjacency transformations ###

def sparse2dense(n_nodes, edges):
    i = torch.cat([edges[0].unsqueeze(0), edges[1].unsqueeze(0)])
    v = torch.ones(i.size(1))
    adj_dense = torch.sparse.FloatTensor(i, v, torch.Size([n_nodes, n_nodes])).to_dense()
    return adj_dense


def dense2networkx(adj_dense):
    g = nx.from_numpy_matrix(adj_dense.detach().to('cpu').numpy(), create_using=nx.Graph)
    return g

### Adjacency transformations ###
#################################

