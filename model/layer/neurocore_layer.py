import torch
import dgl

from satgl.model.sat_model.abstract_model import AbstractSATModel

import torch.nn as nn
from torch.nn.functional import softmax

from ..layer.mlp import MLP
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import node_subgraph

class NeuroCoreLayer(nn.Module):
    """
        NeuroCore Round from NeuroCore.
    """
    def __init__(self, 
                 emb_size, 
                 num_fc):
        super(NeuroCoreLayer, self).__init__()
        self.emb_size = emb_size
        self.num_fc = num_fc

        # update
        self.L_update = MLP(self.emb_size * 3, self.emb_size, self.emb_size, num_layer=num_fc)
        self.C_update = MLP(self.emb_size * 2, self.emb_size, self.emb_size, num_layer=num_fc)
        
        # msg 
        self.L_msg = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.C_msg = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        
        # conv
        self.graph_conv = GraphConv(self.emb_size, self.emb_size, 
                                    allow_zero_in_degree=True, bias=False, 
                                    norm="none", weight=False)

    def flip(self, l_msg, l_pos_index, l_neg_index):
        l_flip_msg = torch.zeros_like(l_msg)
        l_flip_msg[l_pos_index] = l_msg[l_neg_index]
        l_flip_msg[l_neg_index] = l_msg[l_pos_index]
        return l_flip_msg


    def forward(self,
                node_embedding,
                graph):
        num_node = graph.number_of_nodes()
        device = graph.device
        
        # get mask & index
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        l_pos_mask = (node_type == 0).to(device)
        l_neg_mask = (node_type == 1).to(device)
        l_mask = l_pos_mask | l_neg_mask
        c_mask = (node_type == 2).to(device)
        
        l_pos_index = torch.arange(0, num_node).to(device)[l_pos_mask.squeeze(-1)]
        l_neg_index = torch.arange(0, num_node).to(device)[l_neg_mask.squeeze(-1)]
        l_index = torch.arange(0, num_node).to(device)[l_mask.squeeze(-1)]
        c_index = torch.arange(0, num_node).to(device)[c_mask.squeeze(-1)]
        num_node = c_index.shape[0] + l_index.shape[0]

        # clause embedding update 
        l_all_embedding = l_mask * node_embedding
        l_all_msg = self.L_msg(l_all_embedding)
        lc_all_msg = self.graph_conv(graph, l_all_msg)
        lc_msg = lc_all_msg[c_index]
        c_embedding = node_embedding[c_index]
        c_embedding = self.C_update(torch.cat([c_embedding, lc_msg], dim=-1))
        node_embedding[c_index] = c_embedding

        # literal embedding update
        c_all_embedding = c_mask * node_embedding
        c_all_msg = self.C_msg(c_all_embedding)
        cl_all_msg = self.graph_conv(graph, c_all_msg)
        cl_msg = cl_all_msg[l_index]
        l_flip_msg = self.flip(l_all_msg, l_pos_index, l_neg_index)[l_index]
        l_embedding = node_embedding[l_index]
        l_embedding = self.L_update(torch.cat([l_embedding, cl_msg, l_flip_msg], dim=-1))
        node_embedding[l_index] = l_embedding

        return node_embedding

