import os
import subprocess
import tempfile
import shutil
import dgl
import torch

from collections import defaultdict
from tempfile import TemporaryDirectory

cnf2aig_path = os.path.join(os.path.dirname(__file__), "../external/aiger/cnf2aig/cnf2aig")
abc_path = os.path.join(os.path.dirname(__file__), "../external/abc/abc")
aigtoaig_path = os.path.join(os.path.dirname(__file__), "../external/aiger/aiger/aigtoaig")


def parse_aag_file(file_path):
    # read aag
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header = lines[0].strip().split(" ")
    assert header[0] == 'aag', 'The header of AIG file is wrong.'
    # “M”, “I”, “L”, “O”, “A” separated by spaces.
    n_variables = eval(header[1])
    n_inputs = eval(header[2])
    n_outputs = eval(header[4])
    n_and = eval(header[5])
    assert n_outputs == 1, 'The AIG has multiple outputs.'
    assert n_variables == (n_inputs + n_and), 'There are unused AND gates.'
    assert n_variables != n_inputs, '# variable equals to # inputs'
    # Construct AIG graph
    x = []
    edge_index = []
    # node_labels = []
    not_dict = {}

    # Add Literal node
    for i in range(n_inputs):
        x += [0]
        # node_labels += [0]

    # Add AND node
    for i in range(n_inputs + 1, n_inputs + 1 + n_and):
        x += [1]
        # node_labels += [1]

    # sanity-check
    for (i, line) in enumerate(lines[1:1 + n_inputs]):
        literal = line.strip().split(" ")
        assert len(literal) == 1, 'The literal of input should be single.'
        assert int(literal[0]) == 2 * (
                    i + 1), 'The value of a input literal should be the index of variables mutiplying by two.'

    literal = lines[1 + n_inputs].strip().split(" ")[0]
    assert int(literal) == (n_variables * 2) or int(literal) == (
                n_variables * 2) + 1, 'The value of the output literal shoud be (n_variables * 2)'
    sign_final = int(literal) % 2
    index_final_and = int(literal) // 2 - 1

    for (i, line) in enumerate(lines[2 + n_inputs: 2 + n_inputs + n_and]):
        literals = line.strip().split(" ")
        assert len(literals) == 3, 'invalidate the definition of two-input AND gate.'
        assert int(literals[0]) == 2 * (i + 1 + n_inputs)
    var_def = lines[2 + n_variables].strip().split(" ")[0]

    assert var_def == 'i0', 'The definition of variables is wrong.'
    # finish sanity-check

    # Add edge
    for (i, line) in enumerate(lines[2 + n_inputs: 2 + n_inputs + n_and]):
        line = line.strip().split(" ")
        # assert len(line) == 3, 'The length of AND lines should be 3.'
        output_idx = int(line[0]) // 2 - 1
        # assert (int(line[0]) % 2) == 0, 'There is inverter sign in output literal.'

        # 1. First edge
        input1_idx = int(line[1]) // 2 - 1
        sign1_idx = int(line[1]) % 2
        # If there's a NOT node
        if sign1_idx == 1:
            if input1_idx in not_dict.keys():
                not_idx = not_dict[input1_idx]
            else:
                x += [2]
                # node_labels += [2]
                not_idx = len(x) - 1
                not_dict[input1_idx] = not_idx
                edge_index += [[input1_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input1_idx, output_idx]]

        # 2. Second edge
        input2_idx = int(line[2]) // 2 - 1
        sign2_idx = int(line[2]) % 2
        # If there's a NOT node
        if sign2_idx == 1:
            if input2_idx in not_dict.keys():
                not_idx = not_dict[input2_idx]
            else:
                x += [2]
                # node_labels += [2]
                not_idx = len(x) - 1
                not_dict[input2_idx] = not_idx
                edge_index += [[input2_idx, not_idx]]
            edge_index += [[not_idx, output_idx]]
        else:
            edge_index += [[input2_idx, output_idx]]

    if sign_final == 1:
        x += [2]
        # node_labels += [2]
        not_idx = len(x) - 1
        edge_index += [[index_final_and, not_idx]]

    return x, edge_index

def get_node_level(edge_index):
    src_list = [_[0] for _ in edge_index]
    dst_list = [_[1] for _ in edge_index]
    num_node = max(max(src_list), max(dst_list)) + 1
    adj = defaultdict(list)
    for src, dst in zip(src_list, dst_list):
        adj[src].append(dst)
    in_degree = [0 for _ in range(num_node)]
    topo_level = [0 for _ in range(num_node)]

    # init node
    for dst in dst_list:
        in_degree[dst] += 1
    queue = [node for node in range(num_node) if in_degree[node] == 0]


    # topo sort
    while len(queue) > 0:
        u = queue.pop()
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                topo_level[v] = topo_level[u] + 1

    return topo_level

def build_aig(file_path):
    with TemporaryDirectory() as temp_dir:
        # convert cnf to aig
        temp_aig_path = os.path.join(temp_dir, "temp.aig")
        cnf2aig_command = [cnf2aig_path, file_path, temp_aig_path]
        subprocess.run(cnf2aig_command, check=True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # abc optimization
        temp_optimized_aig_path = os.path.join(temp_dir, "temp_optimized.aig")
        abc_command = [
            abc_path,
            "-c",
            "read  %s; balance; print_stats; balance; rewrite -l; rewrite -lz; balance; rewrite -lz; balance; print_stats; cec; write %s" % (temp_aig_path, temp_optimized_aig_path)
        ]
        subprocess.run(abc_command, check=True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # aig to aag
        temp_aag_path = os.path.join(temp_dir, "temp.aag")
        aigtoaig_command = [aigtoaig_path, temp_optimized_aig_path, temp_aag_path]
        subprocess.run(aigtoaig_command, check=True,  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # parse aag file
        node_type, edge_index = parse_aag_file(temp_aag_path)

        # toposort to get node level
        reverse_edge_index = [[_[1], _[0]] for _ in edge_index]
        forward_node_level = get_node_level(edge_index)
        backward_node_level = get_node_level(reverse_edge_index)

        # build graph
        src_list = [_[0] for _ in edge_index]
        dst_list = [_[1] for _ in edge_index]
        g = dgl.graph((src_list, dst_list))
        g.ndata["node_type"] = torch.tensor(node_type).float()
        g.ndata["node_type_one_hot"] = torch.nn.functional.one_hot(torch.tensor(node_type), 3).float() # num_classes should be 3
        g.ndata["forward_node_level"] = torch.tensor(forward_node_level).float()
        g.ndata["backward_node_level"] = torch.tensor(backward_node_level).float()

        return g

if __name__ == "__main__":
    build_aig_graph("../../test.cnf")