from torch import nn
import torch
from models.gcl import GCL, GCL_rf, E_GCL

class AE_parent(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(AE_parent, self).__init__()

    def encode(self, nodes, edges, edge_attr):
        pass

    def decode(self, x):
        pass

    def decode_from_x(self, x, linear_layer=None, C=10, b=-1, remove_diagonal=True):
        n_nodes = x.size(0)
        x_a = x.unsqueeze(0)
        x_b = torch.transpose(x_a, 0, 1)
        X = (x_a - x_b) ** 2
        X = X.view(n_nodes ** 2, -1)
        #X = torch.sigmoid(self.C*torch.sum(X, dim=1) + self.b)
        if linear_layer is not None:
            X = torch.sigmoid(linear_layer(X))
        else:
            X = torch.sigmoid(C*torch.sum(X, dim=1) + b)

        adj_pred = X.view(n_nodes, n_nodes)
        if remove_diagonal:
            adj_pred = adj_pred * (1 - torch.eye(n_nodes).to(self.device))
        return adj_pred

    def forward(self, nodes, edges, edge_attr=None):
        x = self.encode(nodes, edges, edge_attr)
        adj_pred = self.decode(x)
        return adj_pred, x


class AE(AE_parent):
    def __init__(self, hidden_nf, embedding_nf=32, noise_dim=1, device='cpu', act_fn=nn.SiLU(), learnable_dec=1, n_layers=4, attention=0):
        super(AE, self).__init__()
        self.hidden_nf = hidden_nf
        self.embedding_nf = embedding_nf
        self.noise_dim = noise_dim
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.add_module("gcl_0", GCL(max(1, self.noise_dim), self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention, recurrent=False))
        for i in range(1, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=1, act_fn=act_fn, attention=attention))
        self.fc_emb = nn.Linear(self.hidden_nf, self.embedding_nf)

        ### Decoder
        self.fc_dec = None
        if learnable_dec:
            self.fc_dec = nn.Linear(self.embedding_nf, 1)
        self.to(self.device)

    def decode(self, x):
        return self.decode_from_x(x, linear_layer=self.fc_dec)

    def encode(self, nodes, edges, edge_attr=None):
        if self.noise_dim:
            nodes = torch.randn(nodes.size(0), self.noise_dim).to(self.device)
        h, _ = self._modules["gcl_0"](nodes, edges, edge_attr=edge_attr)
        for i in range(1, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
        return self.fc_emb(h)


class AE_rf(AE_parent):
    def __init__(self, embedding_nf=32, nf=64, device='cpu', n_layers=4, act_fn=nn.SiLU(), reg=1e-3, clamp=False):
        super(AE_rf, self).__init__()
        self.embedding_nf = embedding_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.gcl = GCL_rf(nf, reg=reg)
        for i in range(n_layers):
            self.add_module("gcl_%d" % i, GCL_rf(nf, act_fn=act_fn, reg=reg, edge_attr_nf=1, clamp=clamp))

        ### Decoder
        self.w = nn.Parameter(-0.1 * torch.ones(1)).to(device)
        self.b = nn.Parameter(torch.ones(1)).to(device)
        self.to(self.device)

    def decode(self, x):
        return self.decode_from_x(x, C=self.w, b=self.b)

    def encode(self, nodes, edges, edge_attr=None):
        x = torch.randn(nodes.size(0), self.embedding_nf).to(self.device)
        for i in range(0, self.n_layers):
            x, _ = self._modules["gcl_%d" % i](x, edges, edge_attr=edge_attr)
        return x



class AE_EGNN(AE_parent):
    def __init__(self, hidden_nf, K=8, device='cpu', act_fn=nn.SiLU(), n_layers=4, reg = 1e-3, clamp=False):
        super(AE_EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.K = K
        self.device = device
        self.n_layers = n_layers
        self.reg = reg
        ### Encoder
        self.add_module("gcl_0", E_GCL(1, self.hidden_nf, self.hidden_nf, edges_in_d=1, act_fn=act_fn, recurrent=False, clamp=clamp))
        for i in range(1, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=1, act_fn=act_fn, recurrent=True, clamp=clamp))
        #self.fc_emb = nn.Linear(self.hidden_nf, self.embedding_nf)

        ### Decoder
        self.w = nn.Parameter(-0.1*torch.ones(1)).to(device)
        self.b = nn.Parameter(torch.ones(1)).to(device)
        self.to(self.device)

    def decode(self, x):
        return self.decode_from_x(x, C=self.w, b=self.b)

    def encode(self, h, edges, edge_attr=None):
        coords = torch.randn(h.size(0), self.K).to(self.device)
        #h, coords, _ = self._modules["gcl_0"](nodes, edges, coords, edge_attr=edge_attr)
        for i in range(0, self.n_layers):
            h, coords, _ = self._modules["gcl_%d" % i](h, edges, coords, edge_attr=edge_attr)
            coords -= self.reg * coords
            #coords = normalizer(coords)
        return coords


class Baseline(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, device='cpu'):
        super(Baseline, self).__init__()
        self.dummy = nn.Parameter(torch.ones(1))
        self.device = device
        self.to(device)

    def forward(self, nodes, b, c):
        n_nodes = nodes.size(0)
        return torch.zeros(n_nodes, n_nodes).to(self.device) * self.dummy, torch.ones(n_nodes)


def normalizer(x):
    x = x - torch.mean(x, dim=0).unsqueeze(0)
    #x = x / (torch.max(x) - torch.min(x) +1e-8)
    return x