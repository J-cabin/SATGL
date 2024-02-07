import dgl
import os
import torch
import aiger
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from itertools import product

from satgl.data.dataset.abstract_dataset import AbstractDataset
from ..utils import (
    convert_to_datatype,
    aig_calc_level
)


class AIGDataset(AbstractDataset):
    """
        todo
    """
    def __init__(self,
                config,
                processed_data=None):
        super().__init__(config)
        if processed_data is not None:
            self._build_from_processed_data(processed_data)
        else:
            self.name2graph_dict = {}
            self.graph_list = []

            self._build_graph(config)
            self._build_init_feature(config)
            self._load_label(config)
            self._build_graph_list()    
    
    def _build_graph(self, config):
        """
            node 
        """
        graph_list = []
        file_name_list = os.listdir(config.dataset_path)
        for file in file_name_list:
            src_list = []
            dst_list = []
        
            aig = str(aiger.load(os.path.join(config.dataset_path, file))).split('\n')
            header = list(map(eval, aig[0].strip('aag ').split(' ')))
            max_variable = header[0]
            num_input = header[1]
            num_latche = header[2]
            num_output = header[3]
            num_and_gate = header[4]
            body = aig[1 : num_input + num_latche + num_output + num_and_gate + 1]

            # build node index map
            # for pos node, the index is node_idx // 2 - 1
            # for neg node, the index will start from max_variable
            num_neg_node = 0
            node_idx_map = {}
            # pos node map
            for pos_node in range(2, max_variable * 2 + 1, 2):
                node_idx_map[pos_node] = pos_node // 2 - 1
            #neg node map
            for line in body:
                line = list(map(eval, line.split(' ')))
                for node in line:
                    if node % 2 == 1 and node not in node_idx_map:
                        node_idx_map[node] = max_variable + num_neg_node
                        num_neg_node += 1
            
            # build neg edge
            for pos_idx in range(2, max_variable * 2 + 1, 2):
                neg_idx = pos_idx + 1
                if neg_idx in node_idx_map:
                    src_list.append(node_idx_map[pos_idx])
                    dst_list.append(node_idx_map[neg_idx])

            # build edge index
            for line in aig[num_input + 2 : num_input + num_and_gate + 2]:
                # the format is output input1 input2
                line = list(map(eval, line.split(' ')))
                assert(len(line) == 3)

                # build edge
                dst_node = line[0] // 2 - 1
                for input_idx in line[1 : ]:
                    src_list.append(node_idx_map[input_idx])
                    dst_list.append(dst_node)
            
            # build graph
            num_node = max_variable + num_neg_node
            src_list = np.array(src_list)
            dst_list = np.array(dst_list)
            value_list = np.ones_like(src_list)
            csr_data = csr_matrix((value_list, (src_list, dst_list)), shape=(num_node, num_node))
            g = dgl.from_scipy(csr_data)

            # prepare node type
            # input1, input2, ..., inputn, and_gate1, and_gate2, ..., and_gaten, neg_node1, neg_node2, ...
            node_type_list = [0] * num_input + [1] * num_and_gate + [2] * num_neg_node
            g.ndata["node_type"] = torch.tensor(node_type_list).requires_grad_(False)

            info_dict = {}
            info_dict['num_input'] = num_input
            info_dict['num_and_gate'] = num_and_gate
            info_dict['num_neg_node'] = num_neg_node
            info_dict['num_node'] = max_variable + num_neg_node
            info_dict["forward_level"] = torch.tensor(aig_calc_level(np.array([src_list, dst_list])))
            info_dict["backward_level"] = torch.tensor(aig_calc_level(np.array([dst_list, src_list])))
            info_dict["forward_level"].requires_grad = False
            info_dict["backward_level"].requires_grad = False
            self.name2graph_dict[file] = {'g': g, 'info': info_dict}
    def __getitem__(self, index):
        return self.graph_list[index]

    def __len__(self):
        return len(self.graph_list)