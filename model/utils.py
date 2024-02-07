import torch


sat_model = [
    "neurosat",
    "neurocore",
    "nlocalsat",
    "gms",
    "deepsat",
    "querysat",
    "satformer"
]
general_model = [
    "gcn",
    "gat",
    "gin"
]

def lcg_infer_assignment(variable_assgin, graph):
    """Infer the value of some variables given the assignment of other variables
    using local computation graph (lcg).
    Parameters
    ----------
    variable_assgin : dict
        The assignment of some variables.
    graph : dgl.DGLGraph
        The computation graph.
    Returns
    -------
    dict
        The assignment of all variables.
    """
    graph = graph.local_var()
    graph.ndata['var'] = torch.zeros(graph.number_of_nodes(), dtype=torch.bool)
    for var in variable_assgin:
        graph.nodes[var]['var'] = torch.tensor(variable_assgin[var], dtype=torch.bool)
    graph.update_all(lambda edge: {'msg': edge.src['var']}, lambda node: {'var': node.mailbox['msg'].any(dim=1)})
    return {i: bool(graph.nodes[i].data['var']) for i in range(graph.number_of_nodes())}

def get_model(config):
    model = config["model_settings"]["model"]
    if model in sat_model:
        from satgl.model.sat_model.wrapper import (
            LCGWrapper,
            AIGWrapper
        )
        if config["graph_type"] == "lcg":
            return LCGWrapper(config)
        elif config["graph_type"] == "aig":
            return AIGWrapper(config)
    elif model in general_model:
        from satgl.model.general_model.wrapper import (
            LCGWrapper,
            VCGWrapper,
            VIGWrapper,
            LIGWrapper
        )
        if config["graph_type"] == "lcg":
            return LCGWrapper(config)
        elif config["graph_type"] == "vcg":
            return VCGWrapper(config)
        elif config["graph_type"] == "vig":
            return VIGWrapper(config)
        elif config["graph_type"] == "lig":
            return LIGWrapper(config)
    else:
        raise ValueError(f"Model {model} is not supported.")
