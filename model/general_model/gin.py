import dgl
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GINConv
from ..layer.readout_layer import ReadoutLayer
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling



class GIN(nn.Module):
    def __init__(self, config, dataset):
        super(GIN, self).__init__()
        self.layer_list = nn.ModuleList()
        self.batch_norm_list = nn.ModuleList()
        self.linear_prediction_list = nn.ModuleList()
        self.num_fc = config.model_settings["num_fc"]
        self.num_layers = config.model_settings['num_layers']
        self.input_size = dataset.feature_size
        self.hidden_size = config.model_settings['hidden_size']
        self.output_size = config.model_settings['output_size']
        self.dropout_ratio = config.model_settings['dropout_ratio']

        # input linear layer
        self.init = nn.Linear(self.input_size, self.hidden_size)
        
        # build layers
        for layer in range(self.num_layers):
            self.layer_list.append(
                GINConv()
            )
            self.batch_norm_list.append(nn.BatchNorm1d(self.hidden_size))

        # dropout
        self.dropout = nn.Dropout(self.dropout_ratio)

        # pooling
        self.pooling = SumPooling()

        # predict
        self.predict_layer = nn.Linear(self.hidden_size, self.output_size)
        
        # readout 
        self.readout_layer = ReadoutLayer(
            input_size=self.hidden_size,
            output_size=self.output_size,
            pooling=config["model_settings"]["pooling"],
            num_fc=self.num_fc,
            embedding_type="node",
            sigmoid=config["model_settings"]["sigmoid"],
            task_type=config["task_type"]
        )

    def forward(self, graph, embbeding, info):
        num_nodes = graph.number_of_nodes()
        embbeding = self.init(embbeding)
        for layer_id, layer in enumerate(self.layer_list):
            embbeding = layer(graph, embbeding).reshape(num_nodes, -1)
            embbeding = self.batch_norm_list[layer_id](embbeding)
            embbeding = F.relu(embbeding)
        pred = self.readout_layer(graph, embbeding, info)
        return pred
    