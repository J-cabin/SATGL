import torch.nn as nn

from satgl.model.general_model.abstract_model import GeneralAbstractModel
from satgl.model.layer.mlp import MLP
from dgl.nn.pytorch.conv import GATConv


class GAT(GeneralAbstractModel):
    def __init__(self, config):
        super(GAT, self).__init__(config)

    def _build_conv(self):
        self.conv = GATConv(self.hidden_size, self.hidden_size, num_heads=self.config["num_heads"])