import dgl
import os
import torch
import csv
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.nn.functional import one_hot

from scipy.sparse import csr_matrix, save_npz
from itertools import product
from satgl.data.dataset.abstract_dataset import AbstractDataset
from ..utils import convert_to_datatype
from satgl.data.cnf import (
    parse_cnf_file,
    build_lcg,
)
graph_type = ['lig', 'lcg', 'vig', 'vcg', 'nsnet']

class CNFDataset(AbstractDataset):
    """
        CNFDataset build graph to one of the [lig, lcg, vig, vcg]

        Attributes:
            todo
    """
    def __init__(self,
                config,
                processed_data=None):
        super().__init__(config)
        self.config = config

        if processed_data is not None:
            self._build_from_processed_data(processed_data)
        else:
            self._feature_size = 0
            self.name2graph_dict = {}
            self.graph_list = []

            self._build_graph(config)
            # self._build_init_feature(config)
            self._load_label(config)
            self._build_graph_list()


    
    def _build_lcg_graph(self, config):
        """
            lcg graph can be divided into 3 part
                -positive variable
                -negtive variable
                -clause
            if a clause contain a variable then we connect them in the graph
        """
            
        file_name_list = os.listdir(config.dataset_path)
        for file in tqdm(file_name_list):
            num_variable, num_clause, clause_list = parse_cnf_file(os.path.join(config.dataset_path, file))
            g = build_lcg(num_variable, num_clause, clause_list)
            info_dict = {}
            info_dict [ "num_variable"] = num_variable
            self.name2graph_dict[file] = {'g': g, 'info': info_dict}
            
    def _build_nsnet_graph(self, config):
        file_name_list = os.listdir(config.dataset_path)
        for file in tqdm(file_name_list):
            with open(os.path.join(config.dataset_path, file), 'r', encoding='utf-8') as f:
                num_variable = 0
                num_clause = 0
                src_list = []
                dst_list = []
                edge_type = []
                now_clause_id = 0
                for line in f:
                    line = line.strip('\n').strip(" ")
                    if line.startswith('c') or len(line) == 0:
                        continue
                    elif line.startswith('p'):
                        num_variable, num_clause = list(map(eval, line.split(' ')[2:4]))
                    else:
                        clause = list(map(eval, line.split(' ')[:-1]))
                        all_variable = set([abs(v) - 1 for v in clause])
                        for v in all_variable:
                            pos_l_id = v
                            neg_l_id = v + num_variable
                            c_id = num_variable * 2 + now_clause_id

                            # pos literal to clause edge
                            src_list.append(pos_l_id)
                            dst_list.append(c_id)
                            edge_type.append((pos_l_id in all_variable))

                            # neg literal to clause edge
                            src_list.append(neg_l_id)
                            dst_list.append(c_id)
                            edge_type.append((neg_l_id in all_variable))

                            # clause to pos literal edge
                            src_list.append(c_id)
                            dst_list.append(pos_l_id)
                            edge_type.append((pos_l_id in all_variable) + 2)

                            # clause to neg literal edge
                            src_list.append(c_id)
                            dst_list.append(neg_l_id)
                            edge_type.append((neg_l_id in all_variable) + 2)

                        now_clause_id += 1
                num_node = num_variable * 2 + num_clause
                src_list = np.array(src_list)
                dst_list = np.array(dst_list)
                edge_type = np.array(edge_type)
                value_list = np.ones_like(src_list)
                csr_data = csr_matrix((value_list, (src_list, dst_list)), shape=(num_node, num_node))
                g = dgl.from_scipy(csr_data)
                
                # node feature
                node_type = [0] * num_variable + [1] * num_variable + [2] * num_clause
                g.ndata['node_type'] = torch.tensor(node_type)

                # edge feature
                g.edata['edge_type'] = torch.tensor(edge_type)

                # info dict
                info_dict = {}
                info_dict['num_variable'] = num_variable
                info_dict['num_clause'] = num_clause
                self.name2graph_dict[file] = {'g': g, 'info': info_dict}
                

    def _build_lig_graph(self, config):
        """
           lig graph can be divided into 2 part
                -positive variable
                -negtive variable
            if two variable occur in the same clause, we connect them in the graph 
        """
        graph_list = []
        file_name_list = os.listdir(config.dataset_path)
        for file in file_name_list:
            with open(os.path.join(config.dataset_path, file), 'r', encoding='utf-8') as f:
                edge_dict = dict()

                for line in f:
                    line = line.strip('\n').strip(" ")
                    # c -> skip comment
                    # p -> p cnf num_variable num_clause
                    # clause -> [l1, l2, ...]
                    if line.startswith('c') or len(line) == 0:
                        continue
                    elif line.startswith('p'):
                        num_variable, num_clause = list(map(eval, line.split(' ')[2:4]))
                    else:
                        clause = list(map(eval, line.split(' ')[:-1]))
                        clause = [v - 1 if v > 0 else num_variable - v - 1 for v in clause]
                        pr = product(clause, clause)
                        for v1, v2 in pr:
                            edge_dict[(v1, v2)] = 1
                            
            num_node = 2 * num_variable
            dict_key = edge_dict.keys()
            src_list, dst_list = map(list, zip(*dict_key))
            data = list(edge_dict.values())
            src_list = np.array(src_list)
            dst_list = np.array(dst_list)
            value_list = np.ones_like(src_list)
            csr_data = csr_matrix((value_list, (src_list, dst_list)), shape=(num_node, num_node))
            g = dgl.from_scipy(csr_data)
            
            node_type_list = [0] * num_variable + [1] * num_variable
            g.ndata['node_type'] = torch.tensor(node_type_list)

            info_dict = {}
            info_dict['num_variable'] = num_variable
            info_dict['num_clause'] = num_clause
            info_dict["num_node"] = num_node
            self.name2graph_dict[file] = {'g': g, 'info': info_dict}
    
    def _build_vcg_graph(self, config):
        """
        """
        file_name_list = os.listdir(config.dataset_path)
        for file in tqdm(file_name_list):
            node_type_list = []
            with open(os.path.join(config.dataset_path, file), 'r', encoding='utf-8') as f:
                src_list = []
                dst_list = []
                now_clause_id = 0

                for line in f:
                    line = line.strip('\n').strip(" ")
                    # c -> skip comment
                    # p -> p cnf num_variable num_clause
                    # clause -> [l1, l2, ...]
                    if line.startswith('c') or len(line) == 0:
                        continue
                    elif line.startswith('p'):
                        num_variable, num_clause = list(map(eval, line.split(' ')[2:4]))
                    else:
                        clause = list(map(eval, line.split(' ')[:-1]))
                        clause = [abs(v) - 1 for v in clause]
                        src_list.extend(clause)
                        dst_list.extend([now_clause_id + num_variable] * len(clause))
                        now_clause_id += 1
            
            num_node = num_variable + num_clause
            src_list = np.array(src_list)
            dst_list = np.array(dst_list)
            value_list = np.ones_like(src_list)
            csr_data = csr_matrix((value_list, (src_list, dst_list)), shape=(num_node, num_node))
            g = dgl.to_bidirected(dgl.from_scipy(csr_data))
        
            node_type_list = [0] * num_variable + [1] * num_clause
            g.ndata['node_type'] = torch.tensor(node_type_list)

            info_dict = {}
            info_dict['num_variable'] = num_variable
            info_dict['num_clause'] = num_clause
            info_dict["num_node"] = num_node
            self.name2graph_dict[file] = {'g': g, 'info': info_dict}


        
    def _build_vig_graph(self, config):
        """
        """
        graph_list = []
        file_name_list = os.listdir(config.dataset_path)
        for file in file_name_list:
            with open(os.path.join(config.dataset_path, file), 'r', encoding='utf-8') as f:
                edge_dict = dict()

                for line in f:
                    line = line.strip('\n').strip(" ")
                    # c -> skip comment
                    # p -> p cnf num_variable num_clause
                    # clause -> [l1, l2, ...]
                    if line.startswith('c') or len(line) == 0:
                        continue
                    elif line.startswith('p'):
                        num_variable, num_clause = list(map(eval, line.split(' ')[2:4]))
                    else:
                        clause = list(map(eval, line.split(' ')[:-1]))
                        clause = [abs(v) - 1 for v in clause]
                        pr = product(clause, clause)
                        for v1, v2 in pr:
                            edge_dict[(v1, v2)] = 1
                            
            num_node = num_variable
            dict_key = edge_dict.keys()
            src_list, dst_list = map(list, zip(*dict_key))
            data = list(edge_dict.values())
            src_list = np.array(src_list)
            dst_list = np.array(dst_list)
            value_list = np.ones_like(src_list)
            csr_data = csr_matrix((value_list, (src_list, dst_list)), shape=(num_node, num_node))
            g = dgl.from_scipy(csr_data)
            
            node_type_list = [0] * num_variable
            g.ndata['node_type'] = torch.tensor(node_type_list)

            info_dict = {}
            info_dict['num_variable'] = num_variable
            info_dict['num_clause'] = num_clause
            info_dict["num_node"] = num_node
            self.name2graph_dict[file] = {'g': g, 'info': info_dict}
    
    def _build_graph(self, config):
        if config.graph_type in graph_type:
            return getattr(self, f'_build_{config.graph_type}_graph')(config)
        else:
            print(f'{config.graph_type} can not be recognized')
    
    def _build_node_all_zero_feature(self, g):
        return torch.zeros((g.number_of_nodes(), 1))

    def _build_node_all_one_feature(self, g):
        return torch.ones((g.number_of_nodes(), 1))

    def _build_node_random_feature(self, g):
        return torch.rand((g.number_of_nodes(), 1))

    def _build_node_type_feature(self, g):
        unique_values = torch.unique(g.ndata['node_type'])

        feature = one_hot(g.ndata['node_type'], num_classes=len(unique_values))
        return torch.tensor(feature, dtype=torch.float32)
    
    def _build_edge_type_feature(self, g):
        unique_values = torch.unique(g.edata['edge_type'])

        feature = one_hot(g.edata['edge_type'], num_classes=len(unique_values))
        return torch.tensor(feature, dtype=torch.float32)

    def _build_init_feature(self, config):
        node_feature_type = ['all_one', 'all_zero', 'random', 'node_type']
        edge_feature_type = ["edge_type"]
        for item in self.name2graph_dict.values():
            g = item['g']
            if config.feature_type in node_feature_type:
                g.ndata['feature'] = getattr(self, f'_build_{config.feature_type}_feature')(g)
            if config.feature_type in edge_feature_type:
                g.edata['feature'] = getattr(self, f'_build_{config.feature_type}_feature')(g)
    def __getitem__(self, index):
        return self.graph_list[index]

    def __len__(self):
        return len(self.graph_list)