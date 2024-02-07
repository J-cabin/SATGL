import torch
import dgl

from satgl.model.sat_model.abstract_model import AbstractSATModel

import torch.nn as nn
from torch.nn.functional import softmax

from ..layer.mlp import MLP
from ..layer.deepsat_conv import DeepSATConv
from ..layer.readout_layer import ReadoutLayer
from torch.nn import GRU
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import node_subgraph
from dgl import edge_subgraph

class DeepSAT(nn.Module):
    """
        DeepSAT: An EDA-Driven Learning Framework for SAT.
    """
    def __init__(self, config):
        super(DeepSAT, self).__init__()
        self.config = config

        # check config
        if config["graph_type"] not in ["aig"]:
            raise ValueError("DeepSAT only support aig graph.")

        self.device = config.device
        self.hidden_size = config.model_settings["hidden_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.reverse = config.model_settings["reverse"]
        self.num_round = config.model_settings["num_round"]

        # conv
        self.deepsat_conv = DeepSATConv(emb_size=self.hidden_size)

        # update
        self.forward_update = GRU(self.hidden_size + 3, self.hidden_size)
        if self.reverse:
            self.backward_update = GRU(self.hidden_size + 3, self.hidden_size)


    def dag_forward(self, graph, node_embedding):
        # edge index
        forward_edge = torch.stack([graph.edges()[0], graph.edges()[1]], dim=0).to(self.device)
        backward_edge = torch.stack([graph.edges()[1], graph.edges()[0]], dim=0).to(self.device)
        forward_node_level = graph.ndata["forward_node_level"]
        backward_node_level = graph.ndata["backward_node_level"]
        node_type = graph.ndata["node_type_one_hot"]

        # depth
        forward_depth = int(forward_node_level.max().item())
        backward_depth = int(backward_node_level.max().item())

        # forward process
        new_embedding = node_embedding.clone()
        tmp_msg = torch.zeros_like(node_embedding)
        for forward_idx in range(1, forward_depth):
            # get the edge index and node index of the current layer
            layer_mask = (forward_node_level[forward_edge[1]] == forward_idx).to(self.device)
            sub_graph_edge_idx = torch.arange(0, forward_edge.shape[1]).to(self.device)[layer_mask]
            layer_node_idx = torch.unique(forward_edge[1][layer_mask])

            # extract the subgraph
            sub_graph = edge_subgraph(graph, sub_graph_edge_idx, relabel_nodes=True)
            sub_embedding = new_embedding[sub_graph.ndata[dgl.NID]]

            # deepsat conv
            sub_msg = self.deepsat_conv(sub_graph, sub_embedding)
            tmp_msg[sub_graph.ndata[dgl.NID]] = sub_msg
            layer_msg = tmp_msg[layer_node_idx]
            layer_node_type = node_type[layer_node_idx]
            layer_embedding = new_embedding[layer_node_idx]

            # update 
            _, layer_embedding = self.forward_update(
                torch.cat([layer_msg, layer_node_type], dim=1).unsqueeze(0),
                layer_embedding.unsqueeze(0)
            )
            new_embedding[layer_node_idx] = layer_embedding.squeeze(0)
        embedding = new_embedding
        
        # backward process
        if self.reverse:
            new_embedding = embedding.clone()
            tmp_msg = torch.zeros_like(embedding)
            for backward_idx in range(1, backward_depth):
                # get the edge index and node index of the current layer
                layer_mask = (backward_depth[backward_edge[1]] == backward_idx).to(self.device)
                sub_graph_edge_idx = torch.arange(0, backward_edge.shape[1]).to(self.device)[layer_mask]
                layer_node_idx = torch.unique(backward_edge[1][layer_mask])

                # extract the subgraph
                sub_graph = edge_subgraph(graph, sub_graph_edge_idx, relabel_nodes=True)
                sub_embedding = new_embedding[sub_graph.ndata[dgl.NID]]

                # deepsat conv
                sub_msg = self.deepsat_conv(sub_graph, sub_embedding)
                tmp_msg[sub_graph.ndata[dgl.NID]] = sub_msg
                layer_msg = tmp_msg[layer_node_idx]
                layer_node_type = node_type[layer_node_idx]
                layer_embedding = new_embedding[layer_node_idx]

                # update 
                _, layer_embedding = self.backward_update(
                    torch.cat([layer_msg, layer_node_type], dim=1).unsqueeze(0),
                    layer_embedding.unsqueeze(0)
                )
                new_embedding[layer_node_idx] = layer_embedding.squeeze(0)
            embedding = new_embedding
        
        return embedding



    def forward(self, graph, node_embedding):
        num_node = graph.number_of_nodes()

        # dag forward
        for round_idx in range(self.num_round):
            node_embedding = self.dag_forward(graph, node_embedding)

        return node_embedding