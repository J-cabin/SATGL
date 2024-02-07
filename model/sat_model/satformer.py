import torch
import dgl

from satgl.model.sat_model.abstract_model import AbstractSATModel

import math
import torch.nn as nn
from torch.nn.functional import softmax

from ..layer.mlp import MLP
from ..layer.neurosat_layer import NeuroSATLayer
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import node_subgraph



def get_pooling(pooling_method):
    str_to_func = {
        "max": nn.MaxPool2d,
        "mean": nn.AvgPool2d
    }
    return str_to_func[pooling_method]


class SATformer(nn.Module):
    """
        SATformer: Transformers for SAT Solving.
    """
    def __init__(self, config, dataset):
        super(SATformer, self).__init__()
        self.config = config

        # check config
        if config["graph_type"] not in ["lcg"]:
            raise ValueError("NeuroSAT only support lcg graph.")

        self.device = config.device
        self.input_size = dataset.feature_size
        self.hidden_size = config.model_settings["hidden_size"]
        self.output_size = config.model_settings["output_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.num_round = config.model_settings["num_round"]
        self.window_size = config.model_settings["window_size"]
        self.hierarchical_level = config.model_settings["hierarchical_level"]
        self.num_head = config.model_settings["num_head"]
        self.pooling = config.model_settings["pooling"]
        
        self.l_msg_mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.c_msg_mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        self.l_update = nn.LSTM(self.hidden_size * 2, self.hidden_size)
        self.c_update = nn.LSTM(self.hidden_size, self.hidden_size)
        
        # attention linear
        self.query_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_linear = nn.Linear(self.hidden_size, self.hidden_size)

        # attention layer
        self.attention_layer_list = nn.ModuleList()
        for layer_idx in range(self.hierarchical_level):
            self.attention_layer_list.append(
                nn.MultiheadAttention(
                    embed_dim = self.hidden_size,
                    num_heads = self.num_head,
                    batch_first = True
                )
            )

    def gnn_forward(self, graph, embedding, info):
        num_node = graph.number_of_nodes()
        num_variable = info["num_variable"].to(self.device)
        
        # get mask & index
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        l_pos_mask = (node_type == 0).to(self.device)
        l_neg_mask = (node_type == 1).to(self.device)
        l_mask = l_pos_mask | l_neg_mask
        c_mask = (node_type == 2).to(self.device)
        
        l_pos_index = torch.arange(0, embedding.shape[0]).to(self.device)[l_pos_mask.squeeze(-1)]
        l_neg_index = torch.arange(0, embedding.shape[0]).to(self.device)[l_neg_mask.squeeze(-1)]
        l_index = torch.arange(0, embedding.shape[0]).to(self.device)[l_mask.squeeze(-1)]
        c_index = torch.arange(0, embedding.shape[0]).to(self.device)[c_mask.squeeze(-1)]

        # init embedding
        l_embedding = self.L_init(embedding[l_index])
        c_embedding = self.C_init(embedding[c_index])

        # init state 
        l_state = (l_embedding.unsqueeze(0), torch.zeros(1, l_embedding.shape[0], self.hidden_size).to(self.device))
        c_state = (c_embedding.unsqueeze(0), torch.zeros(1, c_embedding.shape[0], self.hidden_size).to(self.device))

        for round_idx in enumerate(range(self.num_round)):
            # run one round neurosat forward to update l_state and c_state
            l_state, c_state = self.neurosat_layer(
                l_state = l_state,
                c_state = c_state,
                graph = graph
            )

        # fill in c_embedding and l_embdding to get all node embedding 
        node_embedding = torch.zeros(num_node, self.hidden_size).to(self.device)
        node_embedding[l_index] = l_state[0].squeeze(0)
        node_embedding[c_index] = c_state[0].squeeze(0)
        return node_embedding
    
    def attention_forward(self, graph, embedding):
        unbatched_graph = dgl.unbatch(graph)
        graph_embedding = []
        offset = 0
        
        for graph_idx, sub_graph in enumerate(unbatched_graph):
            num_clause = sub_graph.num_nodes("c")
            cur_embedding = embedding[offset : offset + num_clause]
            emb_size = self.hidden_size
            window_size = self.window_size
            level_embedding = []

            for level in range(self.hierarchical_level):
                # append level embedding
                pre_level_embedding = cur_embedding.unsqueeze(0)
                pooling = get_pooling(self.pooling)(
                    kernel_size = (pre_level_embedding.shape[1], 1)
                )
                pre_level_embedding = pooling(pre_level_embedding).reshape(-1, emb_size)
                level_embedding.append(pre_level_embedding)

                # pad embedding number to the multiple of w
                num_embedding = cur_embedding.shape[0]
                num_group = math.ceil(num_embedding / window_size)
                pad_size = int(num_group * window_size - num_embedding)
                if pad_size < num_embedding:
                    pad_embedding = cur_embedding[:pad_size]
                else:
                    repeat_embedding = cur_embedding.repeat(math.ceil(pad_size / num_embedding), 1)
                    pad_embedding = repeat_embedding[:pad_size]
                cur_embedding = torch.cat([cur_embedding, pad_embedding], dim=0)

                # apply attention to get group embedding
                query_embedding = self.query_linear(cur_embedding).reshape(num_group, self.window_size, emb_size)
                key_embedding = self.key_linear(cur_embedding).reshape(num_group, self.window_size, emb_size)
                value_embedding = self.value_linear(cur_embedding).reshape(num_group, self.window_size, emb_size)
                cur_embedding, _ = self.attention_layer_list[level](
                    query = query_embedding,
                    key = key_embedding,
                    value = value_embedding,
                    need_weights = False
                )

                # for every group, use pooling function to pool them into one embedding
                pooling = get_pooling(self.pooling)(
                    kernel_size = (self.window_size, 1),
                )
                cur_embedding = pooling(cur_embedding).reshape(-1, emb_size)

            # pool final group embedding to one embedding
            cur_embedding = cur_embedding.unsqueeze(0)
            pooling = get_pooling(self.pooling)(
                kernel_size = (cur_embedding.shape[1], 1),
            )
            cur_embedding = pooling(cur_embedding).reshape(-1, emb_size)

            # concat all level embedding and group embedding to readout
            level_embedding.append(cur_embedding)
            all_embedding = torch.cat(level_embedding, dim=1)
            cur_graph_embedding = self.readout(all_embedding)

            # update offset and graph embedding
            offset += num_clause
            graph_embedding.append(cur_embedding)
        
        graph_embedding = torch.cat(graph_embedding, dim=0)

        return graph_embedding

    def forward(self, lcg_graph, l_embedding, c_embedding):
        l_embedding, c_embedding = self.gnn_forward(lcg_graph, l_embedding, c_embedding)
        graph_embedding = self.attention_forward(lcg_graph, c_embedding)
        return graph_embedding