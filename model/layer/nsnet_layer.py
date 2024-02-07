import torch
import dgl
import numpy as np
import torch.nn as nn
import math
from scipy.sparse import csr_matrix, save_npz

from satgl.config.configurator import Config
from satgl.model.utils import flip
from torch_scatter import scatter_logsumexp
from torch_geometric.utils import index_to_mask
from satgl.model.layer.mlp import MLP

class NSNetLayer(nn.Module):
    """
        NSNet Round from NSNet
    """
    def __init__(self, 
                emb_size,
                num_fc):
        super(NSNetLayer, self).__init__()
        self.emb_size = emb_size
        self.num_fc = num_fc

        self.l2c_msg_update = MLP(
            self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc
        )
        self.c2l_msg_update = MLP(
            self.emb_size, self.emb_size, self.emb_size, num_layer=self.num_fc
        )
        self.l2c_merge_update = MLP(
            self.emb_size * 2, self.emb_size, self.emb_size, num_layer=self.num_fc
        )
        
    def edge_flip(self, edge_embedding):
        odd_index = torch.arange(0, edge_embedding.shape[0], 2).to(edge_embedding.device)
        even_index = torch.arange(1, edge_embedding.shape[0], 2).to(edge_embedding.device)
        flip_edge_embedding = torch.empty_like(edge_embedding)
        flip_edge_embedding[odd_index] = edge_embedding[even_index]
        flip_edge_embedding[even_index] = edge_embedding[odd_index]
        return flip_edge_embedding

    def forward(self, 
                graph,
                l2c_index,
                c2l_index,
                edge_embedding):
        device = edge_embedding.device
        num_node = graph.number_of_nodes()
        num_edge = graph.number_of_edges()
        
        edge_index = torch.cat([graph.edges()[0].unsqueeze(-1), 
                                graph.edges()[1].unsqueeze(-1)], dim=-1).to(device)
        aggr_msg = torch.zeros(num_node, self.emb_size).to(device)
        aggr_msg = dgl.ops.u_add_e_sum(graph, aggr_msg, edge_embedding)
        l2c_mask = index_to_mask(l2c_index, num_edge)
        c2l_mask = index_to_mask(c2l_index, num_edge)
        l2c_edge = edge_index[l2c_index]
        c2l_edge = edge_index[c2l_index]
        l2c_src = l2c_edge[:, 0]
        l2c_dst = l2c_edge[:, 1]
        c2l_src = c2l_edge[:, 0]
        c2l_dst = c2l_edge[:, 1]


        
        # l2c msg update
        l2c_msg = self.l2c_msg_update(aggr_msg[l2c_src] - edge_embedding[l2c_dst])
        flip_l2c_msg = self.edge_flip(l2c_msg)
        l2c_join_msg = torch.cat([l2c_msg, flip_l2c_msg], dim=-1)
        l2c_embedding = self.l2c_merge_update(l2c_join_msg)

        # c2l msg update
        # c2l_msg = aggr_msg[c2l_src] - edge_embedding[c2l_dst]
        # cl2_aggr_msg = scatter_logsumexp(c2l_msg, c2l_dst, dim=0)
        # c2l_msg = cl2_aggr_msg[c2l_dst] - c2l_msg
        # c2l_embedding = self.c2l_msg_update(c2l_msg)
        c2l_msg = scatter_logsumexp(edge_embedding[c2l_dst], c2l_src, dim=0)
        c2l_aggr_msg = c2l_msg[c2l_src] - edge_embedding[c2l_dst]
        c2l_embedding = self.c2l_msg_update(c2l_aggr_msg)

        # update edge embedding
        edge_embedding[l2c_index] = l2c_embedding
        edge_embedding[c2l_index] = c2l_embedding

        return edge_embedding