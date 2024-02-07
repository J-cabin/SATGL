import torch
import dgl
import torch.nn as nn

from satgl.model.layer.mlp import MLP
from satgl.model.general_model.gcn import GCN
from satgl.model.general_model.gat import GAT



suport_general_model = {
    "gcn": GCN,
    "gat": GAT,
}

class LCGWrapper(nn.Module):
    def __init__(self, config):
        super(LCGWrapper, self).__init__()
        self.config = config
        self.feature_type = config["feature_type"]
        self.task = config["task"]
        self.sigmoid = config["model_settings"]["sigmoid"]
        self.model = config["model_settings"]["model"]
        self.device = config["device"]
        self.hidden_size = config.model_settings["hidden_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.node_type_dict = {
            "pos_l": 0,
            "neg_l": 1,
            "c" : 2
        }

        # task specific init
        if self.task == "satisfiability":
            self._satisfiability_init()
        elif self.task == "maxsat":
            self._maxsat_init()
        elif self.task == "unsat_core":
            self._unsat_core_init()
        else:
            raise ValueError(f" task not support.")

        # init
        self.node_feature = nn.parameter.Parameter(torch.randn(1, self.hidden_size))
        self.l_init = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_init = nn.Linear(self.hidden_size, self.hidden_size)

        # load model
        if self.model in suport_general_model:
            self.model = suport_general_model[self.model](config)
        else:
            raise ValueError(f" model not support.")

    def _satisfiability_init(self):
        self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.graph_level_forward = self.graph_pooling
    def _maxsat_init(self):
        self.variable_readout = MLP(self.hidden_size * 2, self.hidden_size, 1, num_layer=self.num_fc)
        self.variable_level_forward = self.variable_pooling

    def _unsat_core_init(self):
        self.clause_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.clause_level_forward = self.clause_pooling
    def get_init_embedding(self, g):
        num_nodes = g.number_of_nodes()
        l_index = (g.ndata["node_type"] != self.node_type_dict["c"]).nonzero().squeeze()
        c_index = (g.ndata["node_type"] == self.node_type_dict["c"]).nonzero().squeeze()
        l_embedding = self.node_feature.repeat(l_index.shape[0], 1)
        c_embedding = self.node_feature.repeat(c_index.shape[0], 1)
        l_embedding = self.l_init(l_embedding)
        c_embedding = self.c_init(c_embedding)
        node_embedding = torch.zeros((num_nodes, self.hidden_size)).to(self.device)
        node_embedding[l_index] = l_embedding
        node_embedding[c_index] = c_embedding


        return node_embedding

    def graph_pooling(self, node_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        pos_l_index = (data["g"].ndata["node_type"] == self.node_type_dict["pos_l"]).nonzero().squeeze()
        neg_l_index = (data["g"].ndata["node_type"] == self.node_type_dict["neg_l"]).nonzero().squeeze()
        pos_l_embedding = node_embedding[pos_l_index]
        neg_l_embedding = node_embedding[neg_l_index]

        mean_v_embedding = (pos_l_embedding + neg_l_embedding) / 2
        g_pooling = dgl.ops.segment_reduce(num_variable, mean_v_embedding, reducer="mean").squeeze(-1)
        g_embedding = self.graph_readout(g_pooling).squeeze(-1)
        if self.sigmoid:
            g_embedding = torch.sigmoid(g_embedding)
        return g_embedding

    def variable_pooling(self, node_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        pos_l_index = (data["g"].ndata["node_type"] == self.node_type_dict["pos_l"]).nonzero().squeeze()
        neg_l_index = (data["g"].ndata["node_type"] == self.node_type_dict["neg_l"]).nonzero().squeeze()
        pos_l_embedding = node_embedding[pos_l_index]
        neg_l_embedding = node_embedding[neg_l_index]
        v_embedding = torch.cat([pos_l_embedding, neg_l_embedding], dim=1)

        v_embedding = self.variable_readout(v_embedding).squeeze(-1)
        if self.sigmoid:
            v_embedding = torch.sigmoid(v_embedding)
        return v_embedding

    def clause_pooling(self, node_embedding, data):
        c_index = (data["g"].ndata["node_type"] == self.node_type_dict["c"]).nonzero().squeeze()
        c_embedding = self.clause_readout(node_embedding[c_index]).squeeze(-1)

        c_embedding = self.clause_readout(c_embedding).squeeze(-1)
        if self.sigmoid:
            c_embedding = torch.sigmoid(c_embedding)
        return c_embedding

    def forward(self, data):
        g = data["g"].to(self.device)
        node_embedding = self.get_init_embedding(g)
        node_embedding = self.model(g, node_embedding)

        # readout
        if self.task == "satisfiability":
            return self.graph_level_forward(node_embedding, data)
        elif self.task == "maxsat":
            return self.variable_level_forward(node_embedding, data)
        elif self.task == "unsat_core":
            return self.clause_level_forward(node_embedding, data)


class VCGWrapper(nn.Module):
    def __init__(self, config):
        super(VCGWrapper, self).__init__()
        self.config = config
        self.feature_type = config["feature_type"]
        self.task = config["task"]
        self.sigmoid = config["model_settings"]["sigmoid"]
        self.model = config["model_settings"]["model"]
        self.device = config["device"]
        self.hidden_size = config.model_settings["hidden_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.node_type_dict = {
            "v": 0,
            "c": 1
        }

        # task specific init
        if self.task == "satisfiability":
            self._satisfiability_init()
        elif self.task == "maxsat":
            self._maxsat_init()
        elif self.task == "unsat_core":
            self._unsat_core_init()
        else:
            raise ValueError(f" task not support.")

    def _satisfiability_init(self):
        self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.graph_level_forward = self.graph_pooling
    def _maxsat_init(self):
        self.variable_readout = MLP(self.hidden_size * 2, self.hidden_size, 1, num_layer=self.num_fc)
        self.variable_level_forward = self.variable_pooling

    def _unsat_core_init(self):
        self.clause_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.clause_level_forward = self.clause_pooling
    def get_init_embedding(self, g):
        node_embedding = self.node_feature.repeat(g.number_of_nodes(), 1)
        return node_embedding

    def graph_pooling(self, node_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        v_index = (data["g"].ndata["node_type"] == self.node_type_dict["v"]).nonzero().squeeze()
        v_embedding = node_embedding[v_index]
        g_pooling = dgl.ops.segment_reduce(num_variable, v_embedding, reducer="mean").squeeze(-1)
        g_embedding = self.graph_readout(g_pooling).squeeze(-1)
        if self.sigmoid:
            g_embedding = torch.sigmoid(g_embedding)
        return g_embedding

    def variable_pooling(self, node_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        v_index = (data["g"].ndata["node_type"] == self.node_type_dict["v"]).nonzero().squeeze()
        v_embedding = node_embedding[v_index]
        v_embedding = self.variable_readout(v_embedding).squeeze(-1)
        if self.sigmoid:
            v_embedding = torch.sigmoid(v_embedding)
        return v_embedding

    def clause_pooling(self, l_embedding, c_embedding, data):
        c_index = (data["g"].ndata["node_type"] == self.node_type_dict["c"]).nonzero().squeeze()
        c_embedding = self.clause_readout(c_embedding[c_index]).squeeze(-1)
        c_embedding = self.clause_readout(c_embedding).squeeze(-1)
        if self.sigmoid:
            c_embedding = torch.sigmoid(c_embedding)
        return c_embedding

    def forward(self, data):
        g = data["g"].to(self.device)
        node_embedding = self.get_init_embedding(g)
        node_embedding = self.model(g, node_embedding)

        # readout
        if self.task == "satisfiability":
            return self.graph_level_forward(node_embedding, data)
        elif self.task == "maxsat":
            return self.variable_level_forward(node_embedding, data)
        elif self.task == "unsat_core":
            return self.clause_level_forward(node_embedding, data)

class VIGWrapper(nn.Module):
    def __init__(self, config):
        super(VIGWrapper, self).__init__()
        self.config = config
        self.feature_type = config["feature_type"]
        self.task = config["task"]
        self.sigmoid = config["model_settings"]["sigmoid"]
        self.model = config["model_settings"]["model"]
        self.device = config["device"]
        self.hidden_size = config.model_settings["hidden_size"]
        self.num_fc = config.model_settings["num_fc"]

        # task specific init
        if self.task == "satisfiability":
            self._satisfiability_init()
        elif self.task == "maxsat":
            self._maxsat_init()
        else:
            raise ValueError(f" task not support for VIG.")

    def _satisfiability_init(self):
        self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.graph_level_forward = self.graph_pooling
    def _maxsat_init(self):
        self.variable_readout = MLP(self.hidden_size * 2, self.hidden_size, 1, num_layer=self.num_fc)
        self.variable_level_forward = self.variable_pooling
    def get_init_embedding(self, g):
        node_embedding = self.node_feature.repeat(g.number_of_nodes(), 1)
        return node_embedding

    def graph_pooling(self, node_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        v_embedding = node_embedding
        g_pooling = dgl.ops.segment_reduce(num_variable, v_embedding, reducer="mean").squeeze(-1)
        g_embedding = self.graph_readout(g_pooling).squeeze(-1)
        if self.sigmoid:
            g_embedding = torch.sigmoid(g_embedding)
        return g_embedding

    def variable_pooling(self, node_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        v_embedding = self.variable_readout(node_embedding).squeeze(-1)
        if self.sigmoid:
            v_embedding = torch.sigmoid(v_embedding)
        return v_embedding

    def forward(self, data):
        g = data["g"].to(self.device)
        node_embedding = self.get_init_embedding(g)
        node_embedding = self.model(g, node_embedding)

        # readout
        if self.task == "satisfiability":
            return self.graph_level_forward(node_embedding, data)
        elif self.task == "maxsat":
            return self.variable_level_forward(node_embedding, data)
        else:
            raise ValueError(f" VIG not support current task .")

class LIGWrapper(nn.Module):
    def __init__(self, config):
        super(LIGWrapper, self).__init__()
        self.config = config
        self.feature_type = config["feature_type"]
        self.task = config["task"]
        self.sigmoid = config["model_settings"]["sigmoid"]
        self.model = config["model_settings"]["model"]
        self.device = config["device"]
        self.hidden_size = config.model_settings["hidden_size"]
        self.num_fc = config.model_settings["num_fc"]
        self.node_type_dict = {
            "pos_l": 0,
            "neg_l": 1,
        }

        # task specific init
        if self.task == "satisfiability":
            self._satisfiability_init()
        elif self.task == "maxsat":
            self._maxsat_init()
        else:
            raise ValueError(f" task not support.")

    def _satisfiability_init(self):
        self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.graph_level_forward = self.graph_pooling
    def _maxsat_init(self):
        self.variable_readout = MLP(self.hidden_size * 2, self.hidden_size, 1, num_layer=self.num_fc)
        self.variable_level_forward = self.variable_pooling

    def _unsat_core_init(self):
        self.clause_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.clause_level_forward = self.clause_pooling
    def get_init_embedding(self, g):
        node_embedding = self.node_feature.repeat(g.number_of_nodes(), 1)
        return node_embedding

    def graph_pooling(self, node_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        pos_l_index = (data["g"].ndata["node_type"] == self.node_type_dict["pos_l"]).nonzero().squeeze()
        neg_l_index = (data["g"].ndata["node_type"] == self.node_type_dict["neg_l"]).nonzero().squeeze()
        pos_l_embedding = node_embedding[pos_l_index]
        neg_l_embedding = node_embedding[neg_l_index]

        mean_v_embedding = (pos_l_embedding + neg_l_embedding) / 2
        g_pooling = dgl.ops.segment_reduce(num_variable, mean_v_embedding, reducer="mean").squeeze(-1)
        g_embedding = self.graph_readout(g_pooling).squeeze(-1)
        if self.sigmoid:
            g_embedding = torch.sigmoid(g_embedding)
        return g_embedding

    def variable_pooling(self, node_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        pos_l_index = (data["g"].ndata["node_type"] == self.node_type_dict["pos_l"]).nonzero().squeeze()
        neg_l_index = (data["g"].ndata["node_type"] == self.node_type_dict["neg_l"]).nonzero().squeeze()
        pos_l_embedding = node_embedding[pos_l_index]
        neg_l_embedding = node_embedding[neg_l_index]
        v_embedding = torch.cat([pos_l_embedding, neg_l_embedding], dim=1)

        v_embedding = self.variable_readout(v_embedding).squeeze(-1)
        if self.sigmoid:
            v_embedding = torch.sigmoid(v_embedding)
        return v_embedding


    def forward(self, data):
        g = data["g"].to(self.device)
        node_embedding = self.get_init_embedding(g)
        node_embedding = self.model(g, node_embedding)

        # readout
        if self.task == "satisfiability":
            return self.graph_level_forward(node_embedding, data)
        elif self.task == "maxsat":
            return self.variable_level_forward(node_embedding, data)
        else:
            raise ValueError(f" VIG not support current task .")
