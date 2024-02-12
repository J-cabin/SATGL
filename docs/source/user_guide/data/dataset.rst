.. _data-dataset:

Dataset
==============

Depending on the type of task, you can implement a ``Dataset`` that applies to a specific task or reads a specific data format.
``Dataset`` is mainly divided into two parts: load data and output data. ``Dataset`` should inherit from
the parent class ``SATTaskAbstractDataset``.

Load Data
----------------

In the data loading module you can implement a function ``_load_dataset`` to read the label information and construct a graph of a
specific type. The function to construct the graph can be found in :ref:`data-rawdata`.

An example can be found in the following code:

.. code:: python

    # MaxSATDataset Dataset
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

Output Data
----------------

Since ``Dataset`` inherits ``DGLDataset``, two functions, ``__getitem__`` and ``__len__``, need to be implemented.

An example can be found in the following code:

.. code:: python

    # MaxSATDataset Dataset
    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)



