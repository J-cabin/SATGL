import torch
import dgl
import math
import torch.nn as nn

from satgl.model.layer.mlp import MLP
from satgl.model.sat_model.neurosat import NeuroSAT
from satgl.model.sat_model.neurocore import NeuroCore
from satgl.model.sat_model.gms import GMS
from satgl.model.sat_model.nlocalsat import NLocalSAT
from satgl.model.sat_model.querysat import QuerySAT
from satgl.model.sat_model.deepsat import  DeepSAT
from satgl.model.general_model.gcn import GCN



suport_general_model = {
    "gcn": GCN
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
        if self.feature_type == "one_hot":
            self.input_size = 3
        else:
            self.input_size = 1
            
        # embedding init layer
        self.init_literal_feature = nn.Parameter(torch.randn(1, self.hidden_size)) 
        self.init_clause_feature = nn.Parameter(torch.randn(1, self.hidden_size))
        self.l_init = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_init = nn.Linear(self.hidden_size, self.hidden_size)

        # task specific init
        if self.task == "satisfiability":
            self._satisfiability_init()
        elif self.task == "maxsat":
            self._maxsat_init()
        elif self.task == "unsat_core":
            self._unsat_core_init()
        else:
            raise ValueError(f" task not support.")

        # sat model init
        if self.model == "neurosat":
            self.model = NeuroSAT(config)
        elif self.model == "neurocore":
            self.model = NeuroCore(config)
        elif self.model == "gms":
            self.model = GMS(config)
        elif self.model == "nlocalsat":
            self.model = NLocalSAT(config)
        elif self.model == "satformer":
            self.model = NeuroSAT(config)
        elif self.model == "querysat":
            self.model = QuerySAT(config)
        elif self.model in suport_general_model:
            self.model = suport_general_model[self.model]
        else:
            raise ValueError(f"{self.model} not support.")

    def _satisfiability_init(self):    
         # readout 
        if self.model == "satformer":
            self.hierarchical_level = self.config["model_settings"]["hierarchical_level"]
            self.num_head = self.config["model_settings"]["num_head"]
            self.window_size = self.config["model_settings"]["window_size"]
            self.pooling = self.config["model_settings"]["pooling"]
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
            self.graph_readout = MLP(self.hidden_size * (self.hierarchical_level + 1), self.hidden_size, 1, num_layer=self.num_fc)
            self.graph_level_forward = self.clause_attention_pooling
        else:
            self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
            self.graph_level_forward = self.graph_pooling

    def _maxsat_init(self):
        self.variable_readout = MLP(self.hidden_size * 2, self.hidden_size, 1, num_layer=self.num_fc)
        self.variable_level_forward = self.variable_pooling

    def _unsat_core_init(self):
        self.clause_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.clause_level_forward = self.clause_pooling
    def get_init_embedding(self, g):
        num_literal = g.number_of_nodes("pos_l") + g.number_of_nodes("neg_l")
        num_clause = g.number_of_nodes("c")
        l_feature = self.init_literal_feature.repeat(num_literal, 1)
        c_feature = self.init_clause_feature.repeat(num_clause, 1)
        l_embedding = self.l_init(l_feature)
        c_embedding = self.c_init(c_feature)

        return l_embedding, c_embedding
    
    def graph_pooling(self, l_embedding, c_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        pos_l_embedding, neg_l_embedding = torch.chunk(l_embedding, 2, dim=0)
        mean_v_embedding = (pos_l_embedding + neg_l_embedding) / 2
        g_pooling = dgl.ops.segment_reduce(num_variable, mean_v_embedding, reducer="mean").squeeze(-1)
        g_embedding = self.graph_readout(g_pooling).squeeze(-1)
        if self.sigmoid:
            g_embedding = torch.sigmoid(g_embedding)
        return g_embedding

    def variable_pooling(self, l_embedding, c_embedding, data):
        pos_l_mebedidng, neg_l_embedding = torch.chunk(l_embedding, 2, dim=0)
        v_embedding = torch.cat([pos_l_mebedidng, neg_l_embedding], dim=1)
        v_embedding = self.variable_readout(v_embedding).squeeze(-1)
        if self.sigmoid:
            v_embedding = torch.sigmoid(v_embedding)
        return v_embedding

    def clause_pooling(self, l_embedding, c_embedding, data):
        c_embedding = self.clause_readout(c_embedding).squeeze(-1)
        if self.sigmoid:
            c_embedding = torch.sigmoid(c_embedding)
        return c_embedding

    def clause_attention_pooling(self, l_embedding, c_embedding, data):
        # pooling func
        def get_pooling(pooling_method):
            str_to_func = {
                "max": nn.MaxPool2d,
                "mean": nn.AvgPool2d
            }
            return str_to_func[pooling_method]

        num_variable = data["info"]["num_variable"].to(self.device)
        num_clause = data["info"]["num_clause"].to(self.device)
        num_graph = num_variable.shape[0]
        graph_embedding = []
        offset = 0
        
        # satformer does not support batch processing
        for graph_idx in range(num_graph):
            cur_num_clause = num_clause[graph_idx].item()
            cur_embedding = c_embedding[offset : offset + cur_num_clause]
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
            cur_graph_embedding = self.graph_readout(all_embedding)

            # update offset and graph embedding
            offset += cur_num_clause
            graph_embedding.append(cur_graph_embedding)
        
        graph_embedding = torch.cat(graph_embedding, dim=0).squeeze(-1)
        if self.sigmoid is not None:
            graph_embedding = torch.sigmoid(graph_embedding)

        return graph_embedding
    
    def forward(self, data):
        g = data["g"].to(self.device)
        l_init_embedding, c_init_embedding = self.get_init_embedding(g)
        l_embedding, c_embedding = self.model(g, l_init_embedding, c_init_embedding)

        # readout
        if self.task == "satisfiability":
            return self.graph_level_forward(l_embedding, c_embedding, data)
        elif self.task == "maxsat":
            return self.variable_level_forward(l_embedding, c_embedding, data)
        elif self.task == "unsat_core":
            return self.clause_level_forward(l_embedding, c_embedding, data)


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
        if self.feature_type == "one_hot":
            self.input_size = 3
        else:
            self.input_size = 1
            
        # embedding init layer
        self.init_literal_feature = nn.Parameter(torch.randn(1, self.hidden_size)) 
        self.init_clause_feature = nn.Parameter(torch.randn(1, self.hidden_size))
        self.v_init = nn.Linear(self.hidden_size, self.hidden_size)
        self.c_init = nn.Linear(self.hidden_size, self.hidden_size)

        # task specific init
        if self.task == "satisfiability":
            self._satisfiability_init()
        
        # sat model init
        raise ValueError(f"VCG now have not model to use.")
                        
    def _satisfiability_init(self):    
        self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.graph_level_forward = self.graph_pooling

    def get_init_embedding(self, g):
        num_variable = g.number_of_nodes("v")
        num_clause = g.number_of_nodes("c")
        v_feature = self.init_literal_feature.repeat(num_variable, 1)
        c_feature = self.init_clause_feature.repeat(num_clause, 1)
        v_embedding = self.v_init(v_feature)
        c_embedding = self.c_init(c_feature)
        return v_embedding, c_embedding
    
    def graph_pooling(self, v_embedding, c_embedding, data):
        num_variable = data["info"]["num_variable"].to(self.device)
        g_pooling = dgl.ops.segment_reduce(num_variable, v_embedding, reducer="mean").squeeze(-1)
        g_embedding = self.graph_readout(g_pooling).squeeze(-1)
        if self.sigmoid:
            g_embedding = torch.sigmoid(g_embedding)
        return g_embedding
    
    def forward(self, data):
        g = data["g"].to(self.device)
        v_init_embedding, c_init_embedding = self.get_init_embedding(g)
        v_embedding, c_embedding = self.model(g, v_init_embedding, c_init_embedding)

        # readout
        if self.task == "satisfiability":
            return self.graph_level_forward(v_embedding, c_embedding, data)


class AIGWrapper(nn.Module):
    def __init__(self, config):
        super(AIGWrapper, self).__init__()
        self.config = config
        self.feature_type = config["feature_type"]
        self.task = config["task"]
        self.sigmoid = config["model_settings"]["sigmoid"]
        self.pooling = config["model_settings"]["pooling"]
        self.model = config["model_settings"]["model"]
        self.device = config["device"]
        self.hidden_size = config.model_settings["hidden_size"]
        self.num_fc = config.model_settings["num_fc"]

        # embedding init layer, the aig graph node type contrains 3
        self.init_feature_list = nn.ParameterList()
        self.init_embedding_list = nn.ModuleList()
        for node_type in range(3):
            self.init_feature_list.append(nn.Parameter(torch.randn(1, self.hidden_size)))
            self.init_embedding_list.append(nn.Linear(self.hidden_size, self.hidden_size))


        # task specific init
        if self.task == "satisfiability":
            self._satisfiability_init()
        else:
            raise ValueError(f" task not support.")

        # sat model init
        if self.model == "deepsat":
            self.model = DeepSAT(config)
        else:
            raise ValueError(f"{self.model} not support.")

    def _satisfiability_init(self):
        # readout
        self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.graph_level_forward = self.graph_pooling

    def get_init_embedding(self, g):
        node_type = g.ndata["node_type"]
        num_nodes = g.number_of_nodes()
        num_classes = 3
        node_embedding = torch.zeros((num_nodes, self.hidden_size)).to(self.device)
        for i in range(num_classes):
            node_type_idx = (node_type == i).nonzero().squeeze().to(self.device)
            init_embedding = self.init_embedding_list[i](self.init_feature_list[i].to(self.device))
            node_embedding[node_type_idx] = init_embedding.repeat(node_type_idx.shape[0], 1)
        return node_embedding


    def graph_pooling(self, node_embedding, data):
        g = data["g"]
        out_node_index = (g.ndata["backward_node_level"] == 0).nonzero().squeeze()
        graph_embedding = self.graph_readout(node_embedding[out_node_index]).squeeze()
        if self.sigmoid:
            graph_embedding = torch.sigmoid(graph_embedding)
        return graph_embedding

    def forward(self, data):
        g = data["g"].to(self.device)
        node_embedding = self.get_init_embedding(g)
        node_embedding = self.model(g, node_embedding)

        # readout
        if self.task == "satisfiability":
            return self.graph_level_forward(node_embedding, data)