.. _cus-data:

Customize Data
==================

Customize Dataset
---------------------

First Step: Load Dataset
>>>>>>>>>>>>>>>>>>>>>>>>>>

First, you should construct function ``_load_dataset`` to read labels and the raw file, such as CNF files or AIG files.
Next, construct the ``_build_graph`` function based on the type of graph to be built. Also, you can add multiple functions
to build different types of graphs. ``_build_graph`` function can also be used directly by inheriting from the ``satgl.data.dataset.sat_dataset.SATTaskAbstractDataset``.
More functions can be found in :ref:`data-rawdata`.

Here's an example function of reading label information stored as a csv and reading CNF files to build graphs:

.. code:: python

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

Second Step. Output dataset
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Since the ``SATTaskAbstractDataset`` provided by SATGL inherits from ``dgl.data.dgl_dataset.DGLDataset``, it is necessary to implement two functions,
``__getitem__`` and ``__len__``, to get the data for each batch during training.

An example can be found in the following code:

.. code:: python

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)

**Complete Code**

.. code:: python

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

Customize DataLoader
------------------------

The dataloader provided by SATGL inherits from ``dgl.dataloading.GraphDataLoader``, so the only thing to do to customize your dataloader
is to design the ``collate_fn`` function to pass to ``dgl.dataloading.GraphDataLoader`` based on the dataset.

An example can be found in the following code:

.. code:: python

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

