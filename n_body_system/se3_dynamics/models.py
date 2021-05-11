import sys

import numpy as np
import torch

# from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from .equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from .equivariant_attention.fibers import Fiber
import time

class TFN(nn.Module):
    """SE(3) equivariant GCN"""
    def __init__(self, num_layers: int, atom_feature_size: int, 
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
                 edge_dim: int=4, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels*num_degrees
        self.edge_dim = edge_dim

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks
        print(self.block0)
        print(self.block1)
        print(self.block2)

    def _build_gcn(self, fibers, out_dim):

        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers-1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))


        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))

        return nn.ModuleList(block0), nn.ModuleList(block1), nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        h = h['0'][...,-1]
        for layer in self.block1:
            h = layer(G, h)

        for layer in self.block2:
            h = layer(h)

        return h


class OursTFN(nn.Module):
    """SE(3) equivariant GCN"""
    def __init__(self, num_layers: int,
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, act_fn=nn.ReLU(),
                 edge_dim: int=4, out_types={1: 1}, in_types={0: 1, 1: 1}, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels*num_degrees
        self.edge_dim = edge_dim
        self.act_fn = act_fn

        self.fibers = {'in': Fiber(dictionary=in_types),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(dictionary=out_types)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0 = blocks
        print(self.block0)

    def _build_gcn(self, fibers, out_dim):

        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers-1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim, act_fn=self.act_fn))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers, act_fn=self.act_fn))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim, act_fn=self.act_fn))

        '''
        block1 = [GMaxPooling()]

        block2 = []
        block2.append(nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(nn.ReLU(inplace=True))
        block2.append(nn.Linear(self.num_channels_out, out_dim))
        '''
        return nn.ModuleList(block0)#, nn.ModuleList(block1), nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f'], '1': G.ndata['f1']}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        '''
        h = h['0'][...,-1]
        for layer in self.block1:
            h = layer(G, h)

        for layer in self.block2:
            h = layer(h)
        '''

        return h


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, atom_feature_size: int, 
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4, 
                 edge_dim: int=4, div: float=4, pooling: str='avg', n_heads: int=1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels)}

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        return h


class OurSE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int,
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4,
                 edge_dim: int=4, div: float=1, pooling: str='avg',
                 n_heads: int=1, act_fn=nn.ReLU(),
                 out_types={1: 1}, in_types={0: 1, 1:1}, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.fibers = {'in': Fiber(dictionary=in_types),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(dictionary=out_types)}

        self.Gblock = self._build_gcn(self.fibers, 1)
        # self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        # print(self.FCblock)
        self.counter = 0
        self.scalar_trick = nn.Parameter(torch.ones(1)*0.01)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads, act_fn=self.act_fn, learnable_skip=False))
            Gblock.append(GNormSE3(fibers['mid'], act_fn=self.act_fn))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim, act_fn=self.act_fn))

        # Pooling
        # if self.pooling == 'avg':
        #     Gblock.append(GAvgPooling())
        # elif self.pooling == 'max':
        #     Gblock.append(GMaxPooling())

        # FC layers
        # FCblock = []
        # FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        # FCblock.append(nn.ReLU(inplace=True))
        # FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock)
        # return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        #t1 = time.time()
        basis, r = get_basis_and_r(G, self.num_degrees-1)
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # t2 = time.time()


        self.counter += 1
        # encoder (equivariant layers)
        h = {'0': G.ndata['f'], '1': G.ndata['f1']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # t3 = time.time()

        # for layer in self.FCblock:
        #     h = layer(h)

        # print("Counter %i \t Time get_basis_and_r: %.3f \t Time forward %.3f" % (self.counter, t2 - t1, t3 - t1))

        #
        for key in h:
            #std = torch.mean(h[key] * h[key])
            # std = torch.std(h[key])
            # print("STD %.5f" % (std.item()))
            h[key] = h[key] * self.scalar_trick

        return h
