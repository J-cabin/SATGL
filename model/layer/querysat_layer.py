import torch
import dgl

from satgl.model.sat_model.abstract_model import AbstractSATModel

import torch.nn as nn
from torch.nn.functional import softmax

from ..layer.mlp import MLP
from ..utils import flip
from ..layer.querysat_conv import QuerySATConv
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import node_subgraph
from torch_geometric.nn.norm import PairNorm
from torch.nn import Softplus

class QuerySATLayer(nn.Module):
    """
        QuerySATLayer Round from QuerySAT.
    """
    def __init__(self, 
                emb_size,
                query_num_mlp_layer=2,
                literal_num_mlp_layer=3,
                clause_num_mlp_layer=2,
                pad_size=0,
                residual_weight=0.1):
        super(QuerySATLayer, self).__init__()
        self.emb_size = emb_size
        self.query_num_mlp_layer = query_num_mlp_layer
        self.literal_num_mlp_layer = literal_num_mlp_layer
        self.clause_num_mlp_layer = clause_num_mlp_layer
        self.pad_size = pad_size
        self.residual_weight = residual_weight

        # update
        self.qmlp = MLP(self.emb_size + self.pad_size, self.emb_size, self.emb_size, num_layer=self.query_num_mlp_layer)
        self.cmlp = MLP(self.emb_size * 2, self.emb_size, self.emb_size, num_layer=self.clause_num_mlp_layer)
        self.lmlp = MLP(self.emb_size * 4, self.emb_size, self.emb_size, num_layer=self.literal_num_mlp_layer)
        
        # conv
        self.graph_conv = GraphConv(self.emb_size, self.emb_size, 
                                    allow_zero_in_degree=True, bias=False, 
                                    norm="none", weight=False)
        self.querysat_conv = QuerySATConv()
        
        # sigmoid
        self.sigmoid = nn.Sigmoid()
        
        # norm
        self.norm = PairNorm()

        # softplus
        self.softplus = Softplus()
    def forward(self,
                node_embedding,
                graph):
        device = graph.device
        num_node = graph.number_of_nodes()
        
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        l_pos_mask = (node_type == 0).to(device)
        l_neg_mask = (node_type == 1).to(device)
        l_mask = l_pos_mask | l_neg_mask
        c_mask = (node_type == 2).to(device)
        
        l_pos_index = torch.arange(0, num_node).to(device)[l_pos_mask.squeeze(-1)]
        l_neg_index = torch.arange(0, num_node).to(device)[l_neg_mask.squeeze(-1)]
        l_index = torch.arange(0, num_node).to(device)[l_mask.squeeze(-1)]
        c_index = torch.arange(0, num_node).to(device)[c_mask.squeeze(-1)]
        
        
        # query update
        num_variable = l_pos_index.shape[0]
        pad_embedding = torch.rand((num_variable, self.pad_size)).to(device)
        v_embedding = node_embedding[l_pos_index]
        q_embedding = self.sigmoid(self.qmlp(torch.cat([v_embedding, pad_embedding], dim=-1)))
        
        # clause update
        c_all_msg = torch.zeros(num_node, self.emb_size).to(device)
        c_all_msg[l_pos_index] = self.softplus(q_embedding)
        c_all_msg[l_neg_index] = self.softplus(-q_embedding)
        c_all_msg = self.graph_conv(graph, c_all_msg)
        c_msg = torch.exp(-c_all_msg[c_index])
        new_c_embedding = self.norm(self.cmlp(torch.cat([node_embedding[c_index], c_msg], dim=-1)))
        c_embedding = new_c_embedding + self.residual_weight * node_embedding[c_index]
        
        # grad update 
        e_loss = torch.sum(c_msg)
        q_grad = torch.autograd.grad(e_loss, q_embedding, retain_graph=True)[0].detach()
        
        
        # variable update
        v_all_msg = torch.zeros(num_node, self.emb_size).to(device)
        v_all_msg[c_index] = new_c_embedding
        v_all_msg = self.graph_conv(graph, v_all_msg)
        v_pos_msg = v_all_msg[l_pos_index]
        v_neg_msg = v_all_msg[l_neg_index]
        new_v_embedding = self.norm(self.lmlp(torch.cat([v_embedding, v_pos_msg, v_neg_msg, q_grad], dim=-1)))
        v_embedding = new_v_embedding + self.residual_weight * node_embedding[l_pos_index]

        node_embedding[l_pos_index] = v_embedding
        node_embedding[l_neg_index] = v_embedding
        node_embedding[c_index] = c_embedding
        
        return node_embedding