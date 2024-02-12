.. _cus-tasks:

Customize Task
==================

First Step: Dataset Wrapper
-------------------------------

Depending on the particular task, add functions to load the dataset and build the dataloader in ``satgl.data.wrapper.SATDataWrapper``.

Example of loading dataset in MaxSAT task:

.. code:: python

    def _load_dataset(self):
        if self.config["task"] == "maxsat":
            self._load_maxsat_dataset()
        else:
            raise NotImplementedError(f"task {self.config['task']} not implemented.")

    def _load_maxsat_dataset(self):
        if self.config["load_split_dataset"] == True:
            train_cnf_dir = os.path.join(self.config["dataset_path"], "train")
            valid_cnf_dir = os.path.join(self.config["dataset_path"], "valid")
            test_cnf_dir = os.path.join(self.config["dataset_path"], "test")
            train_label_path = os.path.join(self.config["dataset_path"], "label", "train.csv")
            valid_label_path = os.path.join(self.config["dataset_path"], "label", "valid.csv")
            test_label_path = os.path.join(self.config["dataset_path"], "label", "test.csv")

            self.logger.info("processing train dataset ...")
            self.train_dataset = MaxSATDataset(self.config, train_cnf_dir, train_label_path)
            self.logger.info("processing valid dataset ...")
            self.valid_dataset = MaxSATDataset(self.config, valid_cnf_dir,
                                                         valid_label_path) if os.path.exists(valid_cnf_dir) else None
            self.logger.info("processing test dataset ...")
            self.test_dataset = MaxSATDataset(self.config, test_cnf_dir, test_label_path) if os.path.exists(
                test_cnf_dir) else None
        else:
            raise NotImplementedError("todo ! Not implemented yet.")

Example of building the dataloader in MaxSAT task:

.. code:: python

    def _build_dataloader(self):
        if self.config["task"] == "maxsat":
            self._build_maxsat_dataloader()
        else:
            raise NotImplementedError(f"task {self.config['task']} not implemented.")

    def _build_maxsat_dataloader(self):
        batch_size = self.config["batch_size"]
        self.train_dataloader = MaxSATDataLoader(self.train_dataset, batch_size, shuffle=True)
        self.valid_dataloader = MaxSATDataLoader(self.valid_dataset, batch_size, shuffle=False) if self.valid_dataset is not None else None
        self.test_dataloader = MaxSATDataLoader(self.test_dataset, batch_size, shuffle=False) if self.test_dataset is not None else None

**Complete Code**

.. code:: python

    from satgl.data.dataset.sat_dataset import MaxSATDataset
    from satgl.data.dataloader.sat_dataloader import MaxSATDataLoader

    class SATDataWrapper():
        def __init__(self, config):
            self.config = config
            self.logger = Logger(config, name='sat_data_wrapper')
            self._load_dataset()
            self._build_dataloader()

        def _load_dataset(self):
            if self.config["task"] == "maxsat":
                self._load_maxsat_dataset()
            else:
                raise NotImplementedError(f"task {self.config['task']} not implemented.")

        def _build_dataloader(self):
            if self.config["task"] == "maxsat":
                self._build_maxsat_dataloader()
            else:
                raise NotImplementedError(f"task {self.config['task']} not implemented.")

        def _load_maxsat_dataset(self):
            if self.config["load_split_dataset"] == True:
                train_cnf_dir = os.path.join(self.config["dataset_path"], "train")
                valid_cnf_dir = os.path.join(self.config["dataset_path"], "valid")
                test_cnf_dir = os.path.join(self.config["dataset_path"], "test")
                train_label_path = os.path.join(self.config["dataset_path"], "label", "train.csv")
                valid_label_path = os.path.join(self.config["dataset_path"], "label", "valid.csv")
                test_label_path = os.path.join(self.config["dataset_path"], "label", "test.csv")

                self.logger.info("processing train dataset ...")
                self.train_dataset = MaxSATDataset(self.config, train_cnf_dir, train_label_path)
                self.logger.info("processing valid dataset ...")
                self.valid_dataset = MaxSATDataset(self.config, valid_cnf_dir,
                                                             valid_label_path) if os.path.exists(valid_cnf_dir) else None
                self.logger.info("processing test dataset ...")
                self.test_dataset = MaxSATDataset(self.config, test_cnf_dir, test_label_path) if os.path.exists(
                    test_cnf_dir) else None
            else:
                raise NotImplementedError("todo ! Not implemented yet.")

        def _build_maxsat_dataloader(self):
            batch_size = self.config["batch_size"]
            self.train_dataloader = MaxSATDataLoader(self.train_dataset, batch_size, shuffle=True)
            self.valid_dataloader = MaxSATDataLoader(self.valid_dataset, batch_size, shuffle=False) if self.valid_dataset is not None else None
            self.test_dataloader = MaxSATDataLoader(self.test_dataset, batch_size, shuffle=False) if self.test_dataset is not None else None

Second Step: Model Wrapper
----------------------------

You should choose model wrapper based on the type of graph being constructed and add task-specific initialization methods to it.
Next, just make minor changes to model wrapper.

For example, if the composition mode is AIG and the task is a satisfiability task, you can add an initialization mode as follows:

.. code:: python

    def _satisfiability_init(self):
        # readout
        self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
        self.graph_level_forward = self.graph_pooling

Then make minor changes to the wrapper, and the complete code looks like this

.. code:: python

    class AIGWrapper(nn.Module):
        def __init__(self, config):
            super(AIGWrapper, self).__init__()
            self.config = config
            self.feature_type = config["feature_type"]
            self.task = config["task"]
            self.sigmoid = config["model_settings"]["sigmoid"]
            self.pooling = config["model_settings"]["pooling"]
            self.model = config["model_settings"]["model"]
            self.device = config["device"]
            self.hidden_size = config.model_settings["hidden_size"]
            self.num_fc = config.model_settings["num_fc"]

            # embedding init layer, the aig graph node type contrains 3
            self.init_feature_list = nn.ParameterList()
            self.init_embedding_list = nn.ModuleList()
            for node_type in range(3):
                self.init_feature_list.append(nn.Parameter(torch.randn(1, self.hidden_size)))
                self.init_embedding_list.append(nn.Linear(self.hidden_size, self.hidden_size))

            # task specific init
            if self.task == "satisfiability":
                self._satisfiability_init()
            else:
                raise ValueError(f" task not support.")

            # sat model init
            if self.model == "deepsat":
                self.model = DeepSAT(config)
            else:
                raise ValueError(f"{self.model} not support.")

        def _satisfiability_init(self):
            # readout
            self.graph_readout = MLP(self.hidden_size, self.hidden_size, 1, num_layer=self.num_fc)
            self.graph_level_forward = self.graph_pooling

        def get_init_embedding(self, g):
            node_type = g.ndata["node_type"]
            num_nodes = g.number_of_nodes()
            num_classes = 3
            node_embedding = torch.zeros((num_nodes, self.hidden_size)).to(self.device)
            for i in range(num_classes):
                node_type_idx = (node_type == i).nonzero().squeeze().to(self.device)
                init_embedding = self.init_embedding_list[i](self.init_feature_list[i].to(self.device))
                node_embedding[node_type_idx] = init_embedding.repeat(node_type_idx.shape[0], 1)
            return node_embedding

        def graph_pooling(self, node_embedding, data):
            g = data["g"]
            out_node_index = (g.ndata["backward_node_level"] == 0).nonzero().squeeze()
            graph_embedding = self.graph_readout(node_embedding[out_node_index]).squeeze()
            if self.sigmoid:
                graph_embedding = torch.sigmoid(graph_embedding)
            return graph_embedding

        def forward(self, data):
            g = data["g"].to(self.device)
            node_embedding = self.get_init_embedding(g)
            node_embedding = self.model(g, node_embedding)

            # readout
            if self.task == "satisfiability":
                return self.graph_level_forward(node_embedding, data)

Third Step. Model Wrapper
--------------------------

For detailed configuration parameters and training settings, consult the
:ref:`cus-trainers`.

Forth Step: Evaluate Metric
----------------------------

You can add the evaluation function in ``satgl.metric`` and then call it in your trainer.



