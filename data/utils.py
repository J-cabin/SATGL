import importlib
import random
import dgl
import os
import torch

from copy import deepcopy
from dgl.dataloading import GraphDataLoader
from satgl.data.wrapper import SATDataWrapper

def concat_collate_fn(item_list):
    # item_list is a list of batched data
    # for graph, use dgl.batch to batch them into a larger graph
    # for tensor, concat them
    if(len(item_list) == 0):
        return None 
    item = item_list[0]
    if isinstance(item, dict):
        return {key: concat_collate_fn([d[key] for d in item_list]) for key in item}
    elif isinstance(item, dgl.DGLGraph):
        return dgl.batch(item_list)
    elif isinstance(item, torch.Tensor):
        if item.dim() == 0:
            return torch.stack(item_list)
        else:
            return torch.cat(item_list, dim=0)
    elif isinstance(item, list):
        return [concat_collate_fn([d[i] for d in item_list]) for i in range(len(item))]
    elif isinstance(item, int):
        return torch.tensor(item_list)
    elif isinstance(item, float):
        return torch.tensor(item_list)

def get_dataset_type(config):
    """
        get dataset type from graph
        now support:
            cnf
            aig
            
    """
    graph2dataset = {
        "lcg": "cnf",
        "lig": "cnf",
        "vcg": "cnf",
        "vig": "cnf",
        "nsnet": "cnf",
        "aig": "aig",
    }
    return graph2dataset[config["graph_type"]]


# def get_dataset(config):
#     task_specific_dataset = {
#         "lcg": SatistifiabilityDataset
#     }
    
#     if config["load_split_dataset"] == True:
#         train_cnf_dir = os.path.join(config["dataset_path"], "train")
#         valid_cnf_dir = os.path.join(config["dataset_path"], "valid")
#         test_cnf_dir = os.path.join(config["dataset_path"], "test")
#         train_label_dir = os.path.join(config["label_path"], "train.csv")
#         valid_label_dir = os.path.join(config["label_path"], "valid.csv")
#         test_label_dir = os.path.join(config["label_path"], "test.csv")
        
#         dataset_class = task_specific_dataset[config["graph_type"]]
#         train_dataset = dataset_class(config, train_cnf_dir, train_label_dir)
#         valid_dataset = dataset_class(config, valid_cnf_dir, valid_label_dir) if os.path.exists(valid_cnf_dir) else None
#         test_dataset = dataset_class(config, test_cnf_dir, test_label_dir) if os.path.exists(test_cnf_dir) else None
        
#         return train_dataset, valid_dataset, test_dataset

def get_dataset(config):
    dataset_type = get_dataset_type(config)
    dataset_module = importlib.import_module(f"sat_toolkit.data.dataset.{dataset_type}_dataset")
    dataset_class = getattr(dataset_module, f"{dataset_type.upper()}Dataset")
    
    if config["load_split_dataset"] == False:
        ratio_list = config["split_ratio"]
        if len(ratio_list) != 3:
            raise ValueError("split_ratio should be a list of 3 elements for train, valid and test.")
        if sum(ratio_list) == 0:
            raise ValueError("split_ratio should not be all zeros.")

        # get init dataset
        dataset = dataset_class(config)

        # normalize ratio
        total_ratio = sum(ratio_list)
        ratio_list = [ratio / total_ratio for ratio in ratio_list]

        # split dataset
        idx_list = list(range(len(dataset)))
        random.shuffle(idx_list)
        
        dataset_list = []
        last_part_end_pos = -1
        for part_idx, ratio in enumerate(ratio_list):
            now_part_start_pos = last_part_end_pos + 1
            now_part_end_pos = now_part_start_pos + int(ratio * len(dataset))
            if part_idx == len(ratio_list) - 1:
                now_part_end_pos = len(dataset)
            now_part = []
            for idx in idx_list[now_part_start_pos:now_part_end_pos]:
                now_part.append(dataset[idx])
            dataset_list.append(dataset_class(config, now_part))
            last_part_end_pos = now_part_end_pos - 1
        return dataset_list
    else:
        dataset_dir = config["dataset_path"]
        label_dir = os.path.join(dataset_dir, "label")
        train_dir = os.path.join(dataset_dir, "train")
        valid_dir = os.path.join(dataset_dir, "valid")
        test_dir = os.path.join(dataset_dir, "test")

        # modify config to load split dataset
        # valid and test dataset can be empty
        train_config = config
        train_config["dataset_path"] = train_dir
        train_config["label_path"] = os.path.join(label_dir, "train.csv")
        train_dataset = dataset_class(train_config)

        valid_config = config
        valid_config["dataset_path"] = valid_dir
        valid_config["label_path"] = os.path.join(label_dir, "valid.csv")
        if os.path.exists(valid_dir):
            valid_dataset = dataset_class(valid_config)
        else:
            valid_dataset = dataset_class(config, [])

        test_config = config
        test_config["dataset_path"] = test_dir
        test_config["label_path"] = os.path.join(label_dir, "test.csv")
        
        if os.path.exists(test_dir):
            test_dataset = dataset_class(test_config)
        else:  
            test_dataset = dataset_class(config, [])
        return (train_dataset, valid_dataset, test_dataset)
            
def get_dataloader(config, dataset_list):
    dataloader_list = []
    for dataset in dataset_list:
        dataloader = GraphDataLoader(dataset, batch_size=config.batch_size, collate_fn=concat_collate_fn, shuffle=True)
        dataloader_list.append(dataloader)
    return dataloader_list
    # return [GraphDataLoader(dataset, batch_size=config.batch_size, collate_fn=concat_collate_fn) for dataset in dataset_list]

def convert_to_datatype(data, type):
    """
        type is str, convert data to corresponding data type
    """
    if type == "float":
        return float(data)
    if type == "list":
        return eval(data)
    
def aig_calc_level(edge_index):
    """
        edge_index: (2, num_edge)
        return the level of each node
    """
    num_node = edge_index.max() + 1
    indegree = [0 for _ in range(num_node)]
    topo_level = [0 for _ in range(num_node)]
    for dst in edge_index[1]:
        indegree[dst] += 1
    queue = []

    for node in range(num_node):
        if indegree[node] == 0:
            queue.append(node)
            topo_level[node] = 0

    while len(queue) != 0:
        node = queue.pop(0)
        for dst in edge_index[1]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
                topo_level[dst] = topo_level[node] + 1
    return list(topo_level)

def get_data(config):
    return SATDataWrapper(config)