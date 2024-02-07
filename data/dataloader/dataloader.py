import dgl


class AbstractDataloader(dgl.dataloading.GraphDataLoader):
    def __init__(self, 
                 config, 
                 dataset,
                 **kwargs):
        super().__init__(self, 
                         dataset, 
                         batch_size=config.batch_size,
                         **kwargs
                         )