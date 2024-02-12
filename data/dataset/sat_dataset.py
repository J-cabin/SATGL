# import dgl
# import os
# import torch
# import csv
# import numpy as np
# import pandas as pd
#
# from satgl.data.cnf import (
#     parse_cnf_file,
#     build_hetero_lcg,
#     build_hetero_vcg,
#     build_homo_lcg,
#     build_homo_vcg,
#     build_homo_vig,
#     build_homo_lig,
#
# )
# from satgl.data.aig import (
#     build_aig
# )
#
#
# from tqdm import tqdm
# from torch.nn.functional import one_hot
# from dgl.data.dgl_dataset import DGLDataset

graph_type_to_build_function = {
    "homo_lcg": build_homo_lcg,
    "homo_vcg": build_homo_vcg,
    "homo_vig": build_homo_vig,
    "homo_lig": build_homo_lig,
    "hetero_lcg": build_hetero_lcg,
    "hetero_vcg": build_hetero_vcg,
    "aig": build_aig
}

sat_model = [
    "neurosat",
    "neurocore",
    "nlocalsat",
    "gms",
    "deepsat",
    "querysat",
    "satformer"
]


class SATTaskAbstractDataset(DGLDataset):
    def __init__(self, config, cnf_dir, label_path, name=None):
        super().__init__(name=name)
        self.config = config
        self.cnf_dir = cnf_dir
        self.label_path = label_path

    def _build_graph(self, num_variable, num_clause, clause_list, file_path):
        # satmodel
        if self.config["model_settings"]["model"] in sat_model:
            if self.config["graph_type"] == "lcg":
                return build_hetero_lcg(num_variable, num_clause, clause_list)
            elif self.config["graph_type"] == "vcg":
                return build_hetero_vcg(num_variable, num_clause, clause_list)
            elif self.config["graph_type"] == "aig":
                return build_aig(file_path)
            else:
                raise NotImplementedError(f"this task graph type {self.config['graph_type']} not support.")
        else:
        # general model
            if self.config["graph_type"] == "lcg":
                return build_homo_lcg(num_variable, num_clause, clause_list)
            elif self.config["graph_type"] == "vcg":
                return build_homo_vcg(num_variable, num_clause, clause_list)
            elif self.config["graph_type"] == "vig":
                return build_homo_vig(num_variable, num_clause, clause_list)
            elif self.config["graph_type"] == "lig":
                return build_homo_lig(num_variable, num_clause, clause_list)




    def _get_info(self, num_variable, num_clause, clause_list):
        return {
            "num_variable": num_variable,
            "num_clause": num_clause
        }

    def _load_dataset(self):
        raise NotImplementedError("This is an abstract class.")

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class.")

    def __len__(self):
        raise NotImplementedError("This is an abstract class.")


class  SatistifiabilityDataset(SATTaskAbstractDataset):
    def __init__(self, config, cnf_dir, label_path):
        super().__init__(config, cnf_dir, label_path, name="satisfiability_dataset")
        self._load_dataset()
    
    def _load_dataset(self):
        label_df = pd.read_csv(self.label_path, sep=',')
        
        # sort benchmark by label to balance the label distribution
        label_occur_times = {}
        self.data_list = []
        for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
            name = row['name']
            label = row['satisfiability']
            cnf_path = os.path.join(self.cnf_dir, name)
            num_variable, num_clause, clause_list = parse_cnf_file(cnf_path)
            cnf_graph = self._build_graph(num_variable, num_clause, clause_list, cnf_path)
            info = self._get_info(num_variable, num_clause, clause_list)
            
            if label not in label_occur_times:
                label_occur_times[label] = 0
            label_occur_times[label] += 1
            self.data_list.append({"g": cnf_graph, "label": label, "info": info})
        self.data_list.sort(key=lambda x: x["label"])
        
    
    def __getitem__(self, idx):
        if idx * 2 + 1 < len(self.data_list):
            return [self.data_list[idx * 2], self.data_list[idx * 2 + 1]]
        elif idx * 2 + 1 == len(self.data_list):
            return [self.data_list[idx * 2]]
        else:
            raise IndexError("Index out of range.")

    def __len__(self):
        return (len(self.data_list) + 1) // 2


class MaxSATDataset(SATTaskAbstractDataset):
    def __init__(self, config, cnf_dir, label_path):
        super().__init__(config, cnf_dir, label_path, name="maxsat_dataset")
        self._load_dataset()

    def _load_dataset(self):
        label_df = pd.read_csv(self.label_path, sep=',')
        self.data_list = []
        for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
            name = row['name']
            label = eval(row['maxsat'])
            cnf_path = os.path.join(self.cnf_dir, name)
            num_variable, num_clause, clause_list = parse_cnf_file(cnf_path)
            cnf_graph = self._build_graph(num_variable, num_clause, clause_list, cnf_path)
            info = self._get_info(num_variable, num_clause, clause_list)
            self.data_list.append({"g": cnf_graph, "label": label, "info": info})

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

class UnSATCoreDataset(SATTaskAbstractDataset):
    def __init__(self, config, cnf_dir, label_path):
        super().__init__(config, cnf_dir, label_path, name="unsat_core_dataset")
        self._load_dataset()

    def _load_dataset(self):
        label_df = pd.read_csv(self.label_path, sep=',')
        self.data_list = []
        for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
            name = row['name']
            label = eval(row['unsat_core'])
            cnf_path = os.path.join(self.cnf_dir, name)
            num_variable, num_clause, clause_list = parse_cnf_file(cnf_path)
            cnf_graph = self._build_graph(num_variable, num_clause, clause_list, cnf_path)
            info = self._get_info(num_variable, num_clause, clause_list)
            self.data_list.append({"g": cnf_graph, "label": label, "info": info})

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)