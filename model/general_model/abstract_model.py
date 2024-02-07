import torch
import torch.nn as nn

from satgl.model.layer.mlp import MLP


class GeneralAbstractModel(nn.Module):
    def __init__(self, config):
        super(GeneralAbstractModel, self).__init__()
        self.layer_list = nn.ModuleList()
        self.num_fc = config.model_settings["num_fc"]
        self.num_layers = config.model_settings['num_layers']
        self.hidden_size = config.model_settings['hidden_size']
        self.graph_type = config['graph_type']

        # build network
        self._build_update()
        self._build_conv()
        self._set_forward()

    def _build_conv(self):
        raise NotImplementedError("Subclass must implement this method")

    def _build_update(self):
        if self.graph_type == "lcg":
            self.l_update = MLP(self.hidden_size * 3, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
            self.c_update = MLP(self.hidden_size * 2, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        elif self.graph_type == "vcg":
            self.v_update = MLP(self.hidden_size * 3, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
            self.c_update = MLP(self.hidden_size * 2, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        elif self.graph_type == "vig":
            self.v_update = MLP(self.hidden_size * 3, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        elif self.graph_type == "lig":
            self.l_update = MLP(self.hidden_size * 3, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
        else:
            raise ValueError("Invalid graph type")

    def _set_forward(self):
        if self.graph_type == "lcg":
            self.forward = self.lcg_forward
        elif self.graph_type == "vcg":
            self.forward = self.vcg_forward
        elif self.graph_type == "vig":
            self.forward = self.vig_forward
        elif self.graph_type == "lig":
            self.forward = self.lig_forward
        else:
            raise ValueError("Invalid graph type")

    def lcg_forward(self, g, node_embedding):
        for layer in range(self.num_layers):
            pre_node_embbedding = node_embedding
            node_type = g.ndata['node_type']
            pos_l_index = (node_type == 0).nonzero().squeeze()
            neg_l_index = (node_type == 1).nonzero().squeeze()
            c_index = (node_type == 2).nonzero().squeeze()
            conv_embedding = self.conv(g, node_embedding)

            pos_cat_embedding = torch.cat(
        [conv_embedding[pos_l_index], pre_node_embbedding[pos_l_index], pre_node_embbedding[neg_l_index]],dim=1)
            neg_cat_embedding = torch.cat(
        [conv_embedding[neg_l_index], pre_node_embbedding[neg_l_index], pre_node_embbedding[pos_l_index]], dim=1)
            c_cat_embedding = torch.cat([conv_embedding[c_index], pre_node_embbedding[c_index]], dim=1)
            pos_new_l_embedding = self.l_update(pos_cat_embedding)
            neg_new_l_embedding = self.l_update(neg_cat_embedding)
            pos_new_c_embedding = self.c_update(c_cat_embedding)

            node_embedding[pos_l_index] = pos_new_l_embedding
            node_embedding[neg_l_index] = neg_new_l_embedding
            node_embedding[c_index] = pos_new_c_embedding

        return node_embedding

    def vcg_forward(self, g, node_embedding):
        pass

    def vig_forward(self, g, node_embedding):
        pass

    def lig_forward(self, g, node_embedding):
        pass




