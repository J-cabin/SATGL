import torch
import dgl

from satgl.model.sat_model.abstract_model import AbstractSATModel

import torch.nn as nn
from torch.nn.functional import softmax

from ..layer.mlp import MLP
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import node_subgraph

class GMSLayer(nn.Module):
    """
        GMSLayer Round from GMS-ESFG.
    """
    def __init__(self, 
                 emb_size, 
                 num_fc):
        super(GMSLayer, self).__init__()
        self.emb_size = emb_size
        self.num_fc = num_fc

        # msg
        self.L_msg = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.C_msg_pos = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.C_msg_neg = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.L_msg_pos = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        self.L_msg_neg = MLP(self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc)
        
        

        # update
        self.L_update = nn.LSTM(self.emb_size, self.emb_size)
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
        l_all_msg = torch.zeros(num_node, self.emb_size).to(device)
        l_all_msg[l_index] = l_hidden
        l_all_msg[l_pos_index] = self.L_msg_pos(l_all_msg[l_pos_index])
        l_all_msg[l_neg_index] = self.L_msg_neg(l_all_msg[l_neg_index])
        lc_all_msg = self.graph_conv(graph, l_all_msg)
        lc_msg = (lc_all_msg).index_select(dim=0, index=c_index)
        _, c_state = self.C_update(lc_msg.unsqueeze(0), c_state)

        c_hidden = c_state[0].squeeze(0)
        c_pre_msg_pos = self.C_msg_pos(c_hidden)
        c_pre_msg_neg = self.C_msg_neg(c_hidden)
        c_all_msg_pos = torch.zeros(num_node, self.emb_size).to(device)
        c_all_msg_neg = torch.zeros(num_node, self.emb_size).to(device)
        c_all_msg_pos[c_index] = c_pre_msg_pos
        c_all_msg_neg[c_index] = c_pre_msg_neg
        cl_all_msg_pos = self.graph_conv(graph, c_all_msg_pos)
        cl_all_msg_neg = self.graph_conv(graph, c_all_msg_neg)
        cl_all_msg = cl_all_msg_pos
        cl_all_msg[l_pos_index] = cl_all_msg[l_pos_index] + cl_all_msg_neg[l_neg_index]
        cl_all_msg[l_neg_index] = cl_all_msg[l_pos_index]
        cl_msg = cl_all_msg.index_select(dim=0, index=l_index)
        _, l_state = self.L_update(cl_msg.unsqueeze(0), l_state)

        return l_state, c_state