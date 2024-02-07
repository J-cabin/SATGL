import torch
import dgl
import torch.nn as nn
from torch.nn.functional import softmax
from satgl.model.layer.mlp import MLP
from satgl.model.conv.hetero import HeteroConv

class NeuroCore(nn.Module):
    def __init__(self, config):
        super(NeuroCore, self).__init__()
        self.config = config

         # check config
        if config["graph_type"] not in ["lcg"]:
            raise ValueError("NeuroCore only support lcg graph.")

        self.device = config.device
        self.hidden_size = config.model_settings["hidden_size"]
        self.output_size = config.model_settings["output_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.num_round = config.model_settings["num_round"]
        
        self.l_msg_mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.c_msg_mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.l_update = MLP(self.hidden_size * 3, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.c_update = MLP(self.hidden_size * 2, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        
        self.conv = HeteroConv()
    
    def forward(self, lcg_graph, l_embedding, c_embedding):
        cur_l_embedding = l_embedding
        cur_c_embedding = c_embedding
        
        for round_idx in enumerate(range(self.num_round)):
            pre_l_embedding = cur_l_embedding
            pre_c_embedding = cur_c_embedding
            # literal message passing
            l_msg = self.l_msg_mlp(pre_l_embedding)
            pos_l_msg, neg_l_msg = torch.chunk(l_msg, 2, dim=0)
            pos_l2c_msg = self.conv(lcg_graph, "pos_l", "pos_l2c", "c", pos_l_msg)
            neg_l2c_msg = self.conv(lcg_graph, "neg_l", "neg_l2c", "c", neg_l_msg)
            l2c_msg = pos_l2c_msg + neg_l2c_msg
            
            
            # clause message passing
            c_msg = self.c_msg_mlp(pre_c_embedding)
            pos_c2l_msg = self.conv(lcg_graph, "c", "pos_c2l", "pos_l", c_msg)
            neg_c2l_msg = self.conv(lcg_graph, "c", "neg_c2l", "neg_l", c_msg)
            c2l_msg = torch.cat([pos_c2l_msg, neg_c2l_msg], dim=0)
            pos_l_embedding, neg_l_embedding = torch.chunk(pre_l_embedding, 2, dim=0)
            flip_l_embedding = torch.cat([neg_l_embedding, pos_l_embedding], dim=0)
            
            # update
            cur_c_embedding = self.c_update(torch.cat([l2c_msg, pre_c_embedding], dim=1))
            cur_l_embedding = self.l_update(torch.cat([c2l_msg, pre_l_embedding, flip_l_embedding], dim=1))
        
        return cur_l_embedding, cur_c_embedding