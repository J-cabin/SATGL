.. _data-dataloader:

DataLoader
==================

Our ``DataLoader`` inherits ``dgl.dataloading.GraphCollator`` and ``dgl.dataloading.GraphDataLoader`` from ``DGL``.
You should implement the ``collate_fn`` function according to ``Dataset``.

An example can be found in the following code:

.. code:: python

    import dgl
    import torch
    from dgl.dataloading import GraphCollator, GraphDataLoader


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

    def MaxSATDataLoader(dataset, batch_size, shuffle=False):
        return GraphDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=maxsat_collate_fn,
        )


