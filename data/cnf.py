import os
import dgl
import torch

def parse_cnf_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        tokens = lines[i].strip().split()
        if len(tokens) < 1 or tokens[0] != 'p':
            i += 1
        else:
            break
    
    if i == len(lines):
        return 0, []
    
    header = lines[i].strip().split()
    num_variable = int(header[2])
    num_clause = int(header[3])
    clause_list = []
    for line in lines[i+1:]:
        tokens = line.strip().split()
        if tokens[0] == 'c':
            continue
        
        clause = [int(s) for s in tokens[:-1]]
        clause_list.append(clause)
    return num_variable, num_clause, clause_list

def build_hetero_lcg(num_variable, num_clause, clause_list):
    pos_src_list = []
    pos_dst_list = []
    neg_src_list = []
    neg_dst_list = []
    
    for c_id, c in enumerate(clause_list):
        for v in c:
            if v > 0:
                pos_src_list.append(v - 1)
                pos_dst_list.append(c_id)
            else:
                neg_src_list.append(-v - 1)
                neg_dst_list.append(c_id)
    
    edge_dict = {
        ("pos_l", "pos_l2c", "c"): (pos_src_list, pos_dst_list),
        ("neg_l", "neg_l2c", "c"): (neg_src_list, neg_dst_list),
        ("c", "pos_c2l", "pos_l"): (pos_dst_list, pos_src_list),
        ("c", "neg_c2l", "neg_l"): (neg_dst_list, neg_src_list),
    }
    
    num_node_dict = {
        "pos_l": num_variable,
        "neg_l": num_variable,
        "c": num_clause,
    }
            
    g = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)
    return g

def build_hetero_vcg(num_variable, num_clause, clause_list):
    pos_src_list = []
    pos_dst_list = []
    neg_src_list = []
    neg_dst_list = []
    
    for c_id, c in enumerate(clause_list):
        for v in c:
            if v > 0:
                pos_src_list.append(v - 1)
                pos_dst_list.append(c_id)
            else:
                neg_src_list.append(-v - 1)
                neg_dst_list.append(c_id)
    
    edge_dict = {
        ("v", "v2c", "c"): (pos_src_list + neg_src_list, pos_dst_list + neg_dst_list),
        ("c", "c2v", "v"): (pos_dst_list + neg_dst_list, pos_src_list + neg_src_list),
    }
    
    num_node_dict = {
        "c": num_clause,
        "v": num_variable
    }
            
    g = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)
    return g


def build_homo_lcg(num_variable, num_clause, clause_list):
    num_nodes = num_variable * 2 + num_clause
    src_list = []
    dst_list = []

    for c_id, c in enumerate(clause_list):
        for v in c:
            v_idx = v - 1 if v > 0 else -v - 1 + num_variable
            c_idx = 2 * num_variable + c_id

            src_list.append(v_idx)
            dst_list.append(c_idx)
            src_list.append(c_idx)
            dst_list.append(v_idx)

    g = dgl.graph((src_list, dst_list), num_nodes=num_nodes)
    node_type = []
    for node_id in range(2 * num_variable + num_clause):
        if node_id < num_variable:
            node_type.append(0)
        elif node_id < 2 * num_variable:
            node_type.append(1)
        else:
            node_type.append(2)
    g.ndata["node_type"] = torch.tensor(node_type).float()
    return g


def build_homo_vcg(num_variable, num_clause, clause_list):
    src_list = []
    dst_list = []

    for c_id, c in enumerate(clause_list):
        for v in c:
            if v > 0:
                src_list.append(v - 1)
            else:
                src_list.append(-v - 1)
            dst_list.append(c_id + num_variable)

    g = dgl.graph((src_list, dst_list))
    node_type = []
    for node_id in range(num_variable + num_clause):
        if node_id < num_variable:
            node_type.append(0)
        else:
            node_type.append(1)
    g.ndata["node_type"] = torch.tensor(node_type).float()
    return g

def build_homo_vig(num_variable, num_clause, clause_list):
    src_list = []
    dst_list = []

    for c_id, c in enumerate(clause_list):
        for i in range(len(c)):
            for j in range(i):
                v1 = c[i] - 1 if c[i] > 0 else -c[i] - 1
                v2 = c[j] - 1 if c[j] > 0 else -c[j] - 1
                src_list.append(v1)
                dst_list.append(v2)

    g = dgl.graph((src_list, dst_list))
    return g

def build_homo_lig(num_variable, num_clause, clause_list):
    src_list = []
    dst_list = []

    for c_id, c in enumerate(clause_list):
        for i in range(len(c)):
            for j in range(i):
                v1 = c[i] - 1 if c[i] > 0 else -c[i] - 1 + num_variable
                v2 = c[j] - 1 if c[j] > 0 else -c[j] - 1 + num_variable
                src_list.append(v1)
                dst_list.append(v2)

    g = dgl.graph((src_list, dst_list))
    node_type = []
    for node_id in range(num_variable + num_clause):
        if node_id < num_variable:
            node_type.append(0)
        else:
            node_type.append(1)
    g.ndata["node_type"] = torch.tensor(node_type).float()
    return g