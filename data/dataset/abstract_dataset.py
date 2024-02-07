import dgl
import os
import torch
import math
import numpy as np
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from itertools import product
# from sat_toolkit.data.utils import convert_to_datatype
from torch.nn.functional import one_hot

class AbstractDataset(dgl.data.dgl_dataset.DGLDataset):
    
    def __init__(self, 
                 config,
                 processed_data=None):
        super().__init__(config.dataset_name)
        self.config = config
        if processed_data is not None:
            self._build_from_processed_data(processed_data)
    
    def _build_graph(self, config):
        raise NotImplementedError

    def _split_dataset(self, config):
        raise NotImplementedError
    
    def _build_feature(self, config):
        raise NotImplementedError
    
    def _build_from_processed_data(self, processed_data):
        self.graph_list = processed_data
    
    def _load_label(self, config):
        """
            load label from config.label_path
            label_path is a csv file
            example:
                name,label
                graph1,1
        """
        label_df = pd.read_csv(config.label_path, sep=',')
        for index, row in label_df.iterrows():
            name = row['name']
            for key in self.config["load_field"]:
                if ":" in key:
                    field, data_type = key.split(":")
                else:
                    field = key
                    data_type = "float"
                value = convert_to_datatype(row[field], data_type)
                self.name2graph_dict[name][field] = value
        
    def _build_graph_list(self):
        """
            conver self.name2graph_dict to self.graph_list
        """
        self.graph_list = []
        label_list = []
        for name, item in self.name2graph_dict.items():
            item_copy = item.copy()
            item_copy['name'] = name
            self.graph_list.append(item_copy)
        
        # make each batch has the similar label distribution
        
        
        

    def _build_all_one_feature(self, g):
        return torch.zeros((g.number_of_nodes(), 1))

    def _build_all_zero_feature(self, g):
        return torch.ones((g.number_of_nodes(), 1))

    def _build_random_feature(self, g):
        return torch.rand((g.number_of_nodes(), 1))

    def _build_node_type_feature(self, g):
        unique_values = torch.unique(g.ndata['node_type'])

        feature = one_hot(g.ndata['node_type'], num_classes=len(unique_values))
        return torch.tensor(feature, dtype=torch.float32)

    def _build_init_feature(self, config):
        feature_type = ['all_one', 'all_zero', 'random', 'node_type']
        for item in self.name2graph_dict.values():
            g = item['g']
            g.ndata['feature'] = getattr(self, f'_build_{config.feature_type}_feature')(g)

    def __getitem__(self, index):
        return self.graph_list[index]

    def __len__(self):
        return len(self.graph_list)

    @property
    def feature_size(self):
        return self.graph_list[0]['g'].ndata['feature'].shape[1]