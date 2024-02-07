import torch
import dgl

import torch.nn as nn
from torch.nn.functional import softmax

from ..layer.mlp import MLP
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl import node_subgraph

class ReadoutLayer(nn.Module):
    """
        readout layer for graph and node level
    """
    def __init__(self, 
                 input_size,
                 output_size,
                 pooling="mean",
                 num_fc=3,
                 embedding_type="literal",
                 sigmoid=True,
                 task_type="graph"):
        super(ReadoutLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pooling = pooling
        self.num_fc = num_fc
        self.embedding_type = embedding_type
        self.task_type = task_type
        
        # build linear layers
        if self.embedding_type == "literal" :
            self.l_out = MLP(self.input_size, self.input_size, self.output_size, num_layer=self.num_fc)
        if self.embedding_type == "variable":
            self.v_out = MLP(self.input_size, self.input_size, self.output_size, num_layer=self.num_fc)
        if self.embedding_type == "clause":
            self.c_out = MLP(self.input_size, self.input_size, self.output_size, num_layer=self.num_fc)
        if self.embedding_type == "edge":
            self.e_out = MLP(self.input_size, self.input_size, self.output_size, num_layer=self.num_fc)
        if self.embedding_type == "node":
            self.n_out = MLP(self.input_size, self.input_size, self.output_size, num_layer=self.num_fc)
        
        # sigmoid
        if sigmoid == True:
            self.sigmoid = nn.Sigmoid()
            
    def literal_readout(self, graph, l_embedding, info):
        device = l_embedding.device
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        num_variable = info["num_variable"].to(device)
        l_pos_mask = (node_type == 0).to(device)
        l_neg_mask = (node_type == 1).to(device)
        l_mask = l_pos_mask | l_neg_mask
        c_mask = (node_type == 2).to(device)
        
        l_pos_index = torch.arange(0, node_type.shape[0]).to(device)[l_pos_mask.squeeze(-1)]
        l_neg_index = torch.arange(0, node_type.shape[0]).to(device)[l_neg_mask.squeeze(-1)]
        l_vote = self.l_out(l_embedding)
        
        if self.task_type == "graph":
            l_vote_pooling = dgl.ops.segment_reduce(num_variable * 2, l_vote, reducer=self.pooling)
            pred = l_vote_pooling.squeeze(-1)
        elif self.task_type == "node":
            pair_vote = torch.cat([l_vote[l_pos_index], l_vote[l_neg_index]], dim=-1)
            pred = torch.mean(pair_vote, dim=-1).squeeze(-1)
        else:
            raise ValueError("task type error")
        
        if hasattr(self, "sigmoid"):
            pred = self.sigmoid(pred)
            
        return pred

    def variable_readout(self, graph, v_embedding, info):
        device = v_embedding.device
        num_variable = info["num_variable"].to(device)
        v_vote = self.v_out(v_embedding)
        
        if self.task_type == "graph":
            v_vote_pooling = dgl.ops.segment_reduce(num_variable, v_vote, reducer=self.pooling)
            pred = v_vote_pooling.squeeze(-1)
        elif self.task_type == "node":
            pred = v_vote.squeeze(-1)
        else:
            raise ValueError("task type error")
        
        if hasattr(self, "sigmoid"):
            pred = self.sigmoid(pred)
            
        return pred

    def node_readout(self, graph, n_embedding, info):
        device = n_embedding.device
        num_node = info["num_node"].to(device)
        n_vote = self.n_out(n_embedding)
        
        if self.task_type == "graph":
            n_vote_pooling = dgl.ops.segment_reduce(num_node, n_vote, reducer=self.pooling)
            pred = n_vote_pooling.squeeze(-1)
        elif self.task_type == "node":
            pred = n_vote.squeeze(-1)
        else:
            raise ValueError("task type error")
        
        if hasattr(self, "sigmoid"):
            pred = self.sigmoid(pred)
            
        return pred

    def edge_readout(self, graph, e_embedding, info):
        device = e_embedding.device
        num_node = graph.number_of_nodes()
        num_edge = graph.number_of_edges()
        num_variable = info["num_variable"].to(device)
        edge_index = torch.cat([graph.edges()[0].unsqueeze(-1), 
                                graph.edges()[1].unsqueeze(-1)], dim=-1).to(device)
        # get mask & index
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        edge_type = graph.edata["edge_type"].unsqueeze(-1)
        l_pos_mask = (node_type == 0).to(device)
        l_neg_mask = (node_type == 1).to(device)
        real_l2c_mask = (edge_type == 1).to(device)
        virtual_l2c_mask = (edge_type == 0).to(device)
        real_c2l_mask = (edge_type == 2).to(device)
        virtual_c2l_mask = (edge_type == 3).to(device)
        l_mask = (node_type != 2).to(device)
        c_mask = (node_type == 2).to(device)
        
        real_l2c_index = torch.arange(0, num_edge).to(self.device)[real_l2c_mask.squeeze(-1)]
        virtual_l2c_index = torch.arange(0, num_edge).to(self.device)[virtual_l2c_mask.squeeze(-1)]
        real_c2l_index = torch.arange(0, num_edge).to(self.device)[real_c2l_mask.squeeze(-1)]
        virtual_c2l_index = torch.arange(0, num_edge).to(self.device)[virtual_c2l_mask.squeeze(-1)]
        l_index = torch.arange(0, num_node).to(self.device)[l_mask.squeeze(-1)]
        c_index = torch.arange(0, num_node).to(self.device)[c_mask.squeeze(-1)]
        l_pos_index = torch.arange(0, num_node).to(self.device)[l_pos_mask.squeeze(-1)]
        l_neg_index = torch.arange(0, num_node).to(self.device)[l_neg_mask.squeeze(-1)]
        
        # aggr
        aggr_embedding = torch.zeros(num_node, self.input_size).to(device)
        aggr_embedding = dgl.ops.u_add_e_sum(graph, aggr_embedding, e_embedding)
        l_embedding = aggr_embedding[l_index]
        
        if self.task_type == "graph":
            l_vote_mean = dgl.ops.segment_reduce(num_variable * 2, l_embedding, reducer=self.pooling)
            pred = self.sigmoid(self.readout(l_vote_mean).squeeze(-1))
        elif self.task_type == "node":
            pred = l_embedding.squeeze(-1)
        
        if hasattr(self, "sigmoid"):
            pred = self.sigmoid(pred)
            
        return pred
    
    def forward(self, graph, embedding, info):
        forward_func = getattr(self, self.embedding_type + "_readout")
        return forward_func(graph, embedding, info)
    
    
class TaskReadoutLayer(nn.Module):
    """
        TaskReadoutLayer layer for:
            -- graph level
            -- variable 
            -- clause
            -- literal
            -- node
    """
    def __init__(self, 
                 input_size,
                 output_size,
                 pooling="mean",
                 num_fc=3,
                 graph_type="lcg",
                 task_type="graph",
                 sigmoid=True):
        super(TaskReadoutLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pooling = pooling
        self.num_fc = num_fc
        self.graph_type = graph_type
        self.task_type = task_type
        self.use_sigmoid = sigmoid
        
        # build network
        self.build_network()

        
    
    def build_network(self):
        if self.graph_type == "lcg":
            if self.task_type == "graph":
                self.readout = MLP(self.input_size * 2, self.input_size, self.output_size, num_layer=self.num_fc)
            if self.task_type == "node":
                self.readout = MLP(self.input_size, self.input_size, self.output_size, num_layer=self.num_fc)
            if self.task_type == "variable":
                self.readout = MLP(self.input_size * 2, self.input_size, self.output_size, num_layer=self.num_fc)
            if self.task_type == "clause":
                self.readout = MLP(self.input_size, self.input_size, self.output_size, num_layer=self.num_fc)
        
        if self.use_sigmoid == True:
            self.sigmoid = nn.Sigmoid()
    
    def lcg_to_graph_readout(self, graph, embedding, info):
        device = embedding.device
        num_variable = info["num_variable"].to(device)
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        l_pos_mask = (node_type == 0).to(device)
        l_neg_mask = (node_type == 1).to(device)
        l_mask = l_pos_mask | l_neg_mask
        
        l_pos_index = torch.arange(0, node_type.shape[0]).to(device)[l_pos_mask.squeeze(-1)]
        l_neg_index = torch.arange(0, node_type.shape[0]).to(device)[l_neg_mask.squeeze(-1)]
        
        v_embedding = torch.cat([embedding[l_pos_index], embedding[l_neg_index]], dim=-1)
        v_embedding = self.readout(v_embedding)
        v_vote_pooling = dgl.ops.segment_reduce(num_variable, v_embedding, reducer=self.pooling)
        pred = v_vote_pooling.squeeze(-1)
        
        if hasattr(self, "sigmoid"):
            pred = self.sigmoid(pred)
            
        return pred
    
    def lcg_to_variable_readout(self, graph, embedding, info):
        device = embedding.device
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        l_pos_mask = (node_type == 0).to(device)
        l_neg_mask = (node_type == 1).to(device)
        l_mask = l_pos_mask | l_neg_mask
        
        l_pos_index = torch.arange(0, node_type.shape[0]).to(device)[l_pos_mask.squeeze(-1)]
        l_neg_index = torch.arange(0, node_type.shape[0]).to(device)[l_neg_mask.squeeze(-1)]
        
        v_embedding = torch.cat([embedding[l_pos_index], embedding[l_neg_index]], dim=-1)
        v_embedding = self.readout(v_embedding)
        pred = v_embedding.squeeze(-1)
        
        if hasattr(self, "sigmoid"):
            pred = self.sigmoid(pred)
            
        return pred

    def lcg_to_clause_readout(self, graph, embedding, info):
        device = embedding.device
        node_type = graph.ndata["node_type"].unsqueeze(-1)
        c_mask = (node_type == 2).to(self.device)
        c_index = torch.arange(0, embedding.shape[0]).to(self.device)[c_mask.squeeze(-1)]
        
        c_embedding = self.readout(embedding[c_index])
        pred = c_embedding.squeeze(-1)
        
        if hasattr(self, "sigmoid"):
            pred = self.sigmoid(pred)
        
        return pred
    
    def lcg_to_node_readout(self, graph, embedding, info):
        n_embedding = self.readout(embedding)
        pred = n_embedding.squeeze(-1)
        
        if hasattr(self, "sigmoid"):
            pred = self.sigmoid(pred)
        
        return pred
            
    
    def forward(self, graph, embedding, info):
        forward_func = getattr(self, self.graph_type + "_to_" + self.task_type + "_readout")
        return forward_func(graph, embedding, info)