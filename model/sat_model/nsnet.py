import torch
import dgl
import numpy as np
import torch.nn as nn
import math
from scipy.sparse import csr_matrix, save_npz

from satgl.config.configurator import Config
from satgl.model.utils import flip
from satgl.model.layer.mlp import MLP
from satgl.model.layer.nsnet_layer import NSNetLayer

class NSNet(nn.Module):
    """
        from NSNet: A General Neural Probabilistic Framework for Satisfiability Problems
    """

    def __init__(self, config, dataset=None):
        super(NSNet, self).__init__()
        self.device = config.device
        self.input_size = config.model_settings["input_size"]
        self.hidden_size = config.model_settings["hidden_size"]
        self.output_size = config.model_settings["output_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.num_round = config.model_settings["num_round"]

        # init 
        self.l2c_init = nn.Linear(self.input_size, self.hidden_size)
        self.c2l_init = nn.Linear(self.input_size, self.hidden_size)

        # demon
        self.demon = math.sqrt(self.hidden_size)

        # update 
        self.l2c_msg_update = MLP(
            self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc
        )
        self.c2l_msg_update = MLP(
            self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc
        )
        self.l2c_merge_update = MLP(
            self.hidden_size * 2, self.hidden_size, self.hidden_size, num_layer=self.num_fc
        )

        # softmax 
        self.softmax = nn.Softmax(dim=1)

        # layer
        self.nsnet_layer = NSNetLayer(emb_size=self.hidden_size, num_fc=self.num_fc)
        
        # readout
        self.readout = MLP(
            self.hidden_size, self.hidden_size, self.output_size, num_layer=self.num_fc
        )
        
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def graph_forward(self, graph, embedding, info):
        num_variable = info["num_variable"].to(self.device)
        num_clause = info["num_clause"].to(self.device)
        num_node = graph.number_of_nodes()
        num_edge = graph.number_of_edges()

        # get mask & index
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        edge_type = graph.edata["edge_type"].unsqueeze(-1)
        l_pos_mask = (node_type == 0).to(self.device)
        l_neg_mask = (node_type == 1).to(self.device)
        real_l2c_mask = (edge_type == 1).to(self.device)
        virtual_l2c_mask = (edge_type == 0).to(self.device)
        real_c2l_mask = (edge_type == 2).to(self.device)
        virtual_c2l_mask = (edge_type == 3).to(self.device)
        l_mask = (node_type != 2).to(self.device)
        c_mask = (node_type == 2).to(self.device)

        l2c_mask = real_l2c_mask | virtual_l2c_mask
        c2l_mask = real_c2l_mask | virtual_c2l_mask

        real_l2c_index = torch.arange(0, num_edge).to(self.device)[real_l2c_mask.squeeze(-1)]
        virtual_l2c_index = torch.arange(0, num_edge).to(self.device)[virtual_l2c_mask.squeeze(-1)]
        real_c2l_index = torch.arange(0, num_edge).to(self.device)[real_c2l_mask.squeeze(-1)]
        virtual_c2l_index = torch.arange(0, num_edge).to(self.device)[virtual_c2l_mask.squeeze(-1)]
        l_index = torch.arange(0, num_node).to(self.device)[l_mask.squeeze(-1)]
        c_index = torch.arange(0, num_node).to(self.device)[c_mask.squeeze(-1)]
        l_pos_index = torch.arange(0, num_node).to(self.device)[l_pos_mask.squeeze(-1)]
        l_neg_index = torch.arange(0, num_node).to(self.device)[l_neg_mask.squeeze(-1)]

        
        l2c_index = torch.arange(0, num_edge).to(self.device)[l2c_mask.squeeze(-1)]
        c2l_index = torch.arange(0, num_edge).to(self.device)[c2l_mask.squeeze(-1)]

        # init
        l2c_embedding = self.l2c_init(embedding[l2c_index])
        c2l_embedding = self.c2l_init(embedding[c2l_index])
        edge_embedding = torch.empty(num_edge, self.hidden_size).to(self.device)
        edge_embedding[l2c_index] = l2c_embedding
        edge_embedding[c2l_index] = c2l_embedding

        for round_idx in range(self.num_round):
            edge_embedding = self.nsnet_layer(
                graph,
                l2c_index,
                c2l_index,
                edge_embedding
            )

        # readout
        edge_index = torch.cat([graph.edges()[0].unsqueeze(-1), 
                                graph.edges()[1].unsqueeze(-1)], dim=-1).to(self.device)
        aggr_embedding = torch.zeros(num_node, self.hidden_size).to(self.device)
        aggr_embedding = dgl.ops.u_add_e_sum(graph, aggr_embedding, edge_embedding)
        l_embedding = aggr_embedding[l_index]
        l_vote_mean = dgl.ops.segment_reduce(num_variable * 2, l_embedding, reducer="mean")
        pred = self.sigmoid(self.readout(l_vote_mean).squeeze(-1))
        
        return pred