import dgl
import torch
import itertools
from dgl.dataloading import GraphCollator, GraphDataLoader



def satisfiability_collate_fn(data):
    return GraphCollator().collate([item for item in list(itertools.chain(*data))])

def maxsat_collate_fn(data):
    if (len(data) == 0):
        return None
    elem = data[0]
    if isinstance(elem, dict):
        return {key: maxsat_collate_fn([d[key] for d in data]) for key in elem}
    elif isinstance(elem, dgl.DGLGraph):
        return dgl.batch(data)
    elif isinstance(elem, torch.Tensor):
        if elem.dim() == 0:
            return torch.stack(data)
        else:
            return torch.cat(data, dim=0)
    elif isinstance(elem, list):
        return torch.cat([maxsat_collate_fn(e) for e in data])
    elif isinstance(elem, int):
        return torch.tensor(data)
    elif isinstance(elem, float):
        return torch.tensor(data)

def unsat_core_collate_fn(data):
    if (len(data) == 0):
        return None
    batched_data = {
        "g": GraphCollator().collate([elem["g"] for elem in data]),
        "label": torch.cat([torch.tensor(elem["label"]).float() for elem in data]),
        "info": GraphCollator().collate([elem["info"] for elem in data])
    }
    return batched_data


def SatistifiabilityDataLoader(dataset, batch_size, shuffle=False):
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=satisfiability_collate_fn,
    )


def MaxSATDataLoader(dataset, batch_size, shuffle=False):
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=maxsat_collate_fn,
    )

def UnsatCoreDataLoader(dataset, batch_size, shuffle=False):
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=unsat_core_collate_fn,
    )
