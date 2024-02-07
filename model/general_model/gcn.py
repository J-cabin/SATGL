import torch.nn as nn

from satgl.model.general_model.abstract_model import GeneralAbstractModel
from satgl.model.layer.mlp import MLP
from dgl.nn.pytorch.conv import GraphConv



class GCN(GeneralAbstractModel):
    def __init__(self, config):
        super(GCN, self).__init__(config)

    def _build_conv(self):
        self.conv = GraphConv(self.hidden_size, self.hidden_size, allow_zero_in_degree=True, norm="none")

    