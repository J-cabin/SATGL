import torch
import dgl
import torch.nn as nn
from torch.nn.functional import softmax
from satgl.model.layer.mlp import MLP
from satgl.model.conv.hetero import HeteroConv
from torch_geometric.nn.norm import PairNorm


class QuerySAT(nn.Module):
    """
        QuerySAT
    """
    def __init__(self, config):
        super(QuerySAT, self).__init__()
        self.config = config

        # check config
        if config["graph_type"] not in ["lcg"]:
            raise ValueError("QuerySAT only support lcg graph.")

        self.device = config.device
        self.hidden_size = config.model_settings["hidden_size"]
        self.output_size = config.model_settings["output_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.pad_size = config.model_settings["pad_size"]
        self.num_round = config.model_settings["num_round"]
        
        self.q_mlp = MLP(self.hidden_size * 2 + self.pad_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        # self.c_mlp = MLP(self.hidden_size * 2, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.v_update = MLP(self.hidden_size * 4, self.hidden_size * 2, self.hidden_size * 2, num_layer=self.num_fc)
        self.l_update = MLP(self.hidden_size * 3, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.c_update = MLP(self.hidden_size * 3, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.softplus = nn.Softplus()
        
        self.conv = HeteroConv()
        self.norm = PairNorm()

    
    def forward(self, lcg_graph, l_embedding, c_embedding):
        num_variable = l_embedding.shape[0] // 2
        for round_idx in enumerate(range(self.num_round)):
            # get query and clause loss
            v_embedding = torch.cat(torch.chunk(l_embedding, 2, dim=0), dim=1)
            noise_embedding = torch.randn((num_variable, self.pad_size)).to(self.device)
            q_embedding = self.q_mlp(torch.cat([v_embedding, noise_embedding], dim=1))
            pos_q_msg = self.softplus(q_embedding)
            neg_q_msg = self.softplus(-q_embedding)
            pos_q2c_msg = self.conv(lcg_graph, "pos_l", "pos_l2c", "c", pos_q_msg)
            neg_q2c_msg = self.conv(lcg_graph, "neg_l", "neg_l2c", "c", neg_q_msg)
            q2c_msg = pos_q2c_msg + neg_q2c_msg
            e_embedding = torch.exp(-q2c_msg)

            # literal message passing
            l_msg = l_embedding
            pos_l_msg, neg_l_msg = torch.chunk(l_msg, 2, dim=0)
            pos_l2c_msg = self.conv(lcg_graph, "pos_l", "pos_l2c", "c", pos_l_msg)
            neg_l2c_msg = self.conv(lcg_graph, "neg_l", "neg_l2c", "c", neg_l_msg)
            l2c_msg = pos_l2c_msg + neg_l2c_msg

            # clause message passing
            c_msg = c_embedding
            pos_c2l_msg = self.conv(lcg_graph, "c", "pos_c2l", "pos_l", c_msg)
            neg_c2l_msg = self.conv(lcg_graph, "c", "neg_c2l", "neg_l", c_msg)
            c2l_msg = torch.cat([pos_c2l_msg, neg_c2l_msg], dim=0)
            pos_l_embedding, neg_l_embedding = torch.chunk(l_embedding, 2, dim=0)
            flip_l_embedding = torch.cat([neg_l_embedding, pos_l_embedding], dim=0)

            # update
            v_embedding = self.v_update(torch.cat([v_embedding, pos_c2l_msg, neg_c2l_msg], dim=1))
            c_embedding = self.c_update(torch.cat([l2c_msg, c_embedding, e_embedding], dim=1))
            l_embedding = torch.cat(torch.chunk(v_embedding, 2, dim=1), dim=0)

        return l_embedding, c_embedding
