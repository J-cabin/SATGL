import torch
import dgl

from satgl.model.sat_model.abstract_model import AbstractSATModel

import torch.nn as nn
from torch.nn.functional import softmax

from ..layer.mlp import MLP
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import node_subgraph

class NeuroSATLayer(nn.Module):
    """
        NeuroSAT Round from NeuroSAT.
    """
    def __init__(self, 
                 emb_size, 
                 num_fc):
        super(NeuroSATLayer, self).__init__()
        self.emb_size = emb_size
        self.num_fc = num_fc

        # msg
        self.L_msg = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.C_msg = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)

        # update
        self.L_update = nn.LSTM(self.emb_size * 2, self.emb_size)
        self.C_update = nn.LSTM(self.emb_size, self.emb_size)
        
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
                l_state,
                c_state,
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

        l_hidden = l_state[0].squeeze(0)
        l_pre_msg = self.L_msg(l_hidden)
        l_all_msg = torch.zeros(num_node, self.emb_size).to(device)
        l_all_msg[l_index] = l_pre_msg
        lc_all_msg = self.graph_conv(graph, l_all_msg)
        lc_msg = torch.index_select(lc_all_msg, dim=0, index=c_index)
        _, c_state = self.C_update(lc_msg.unsqueeze(0), c_state)

        c_hidden = c_state[0].squeeze(0)
        c_pre_msg = self.C_msg(c_hidden)
        c_all_msg = torch.zeros(num_node, self.emb_size).to(device)
        c_all_msg[c_index] = c_pre_msg
        cl_all_msg = self.graph_conv(graph, c_all_msg)
        cl_msg = torch.index_select(cl_all_msg, dim=0, index=l_index)
        l_flip_msg = self.flip(l_all_msg, l_pos_index, l_neg_index)[l_index]
        _, l_state = self.L_update(torch.cat([cl_msg, l_flip_msg], dim=1).unsqueeze(0), l_state)

        return l_state, c_state