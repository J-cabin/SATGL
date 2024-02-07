import torch
import dgl
import torch.nn as nn
from torch.nn.functional import softmax
from satgl.model.layer.mlp import MLP
from satgl.model.conv.hetero import HeteroConv

class NLocalSAT(nn.Module):
    """
        NLocalSAT: Boosting Local Search with Solution Prediction.
    """
    def __init__(self, config):
        super(NLocalSAT, self).__init__()
        self.config = config

        # check config
        if config["graph_type"] not in ["lcg"]:
            raise ValueError("NLocalSAT only support lcg graph.")

        self.device = config.device
        self.hidden_size = config.model_settings["hidden_size"]
        self.output_size = config.model_settings["output_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.num_round = config.model_settings["num_round"]
        
        self.l_msg_mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.c_msg_mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.l_update = nn.GRUCell(self.hidden_size * 2, self.hidden_size)
        self.c_update = nn.GRUCell(self.hidden_size, self.hidden_size)
        
        self.conv = HeteroConv()
    
    def forward(self, lcg_graph, l_embedding, c_embedding):
        num_literal = l_embedding.shape[0]
        num_clause = c_embedding.shape[0]
        
        for round_idx in enumerate(range(self.num_round)):
            # literal message passing
            l_msg = self.l_msg_mlp(l_embedding)
            pos_l_msg, neg_l_msg = torch.chunk(l_msg, 2, dim=0)
            pos_l2c_msg = self.conv(lcg_graph, "pos_l", "pos_l2c", "c", pos_l_msg)
            neg_l2c_msg = self.conv(lcg_graph, "neg_l", "neg_l2c", "c", neg_l_msg)
            l2c_msg = pos_l2c_msg + neg_l2c_msg
            
            
            # clause message passing
            c_msg = self.c_msg_mlp(c_embedding)
            pos_c2l_msg = self.conv(lcg_graph, "c", "pos_c2l", "pos_l", c_msg)
            neg_c2l_msg = self.conv(lcg_graph, "c", "neg_c2l", "neg_l", c_msg)
            c2l_msg = torch.cat([pos_c2l_msg, neg_c2l_msg], dim=0)
            pos_l_embedding, neg_l_embedding = torch.chunk(l_embedding, 2, dim=0)
            flip_l_hidden = torch.cat([neg_l_embedding, pos_l_embedding], dim=0)
            
            # update
            c_embedding = self.c_update(l2c_msg, c_embedding)
            l_embedding = self.l_update(torch.cat([c2l_msg, flip_l_hidden], dim=1), l_embedding)


        return l_embedding, c_embedding