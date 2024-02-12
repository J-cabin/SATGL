Config Settings
>>>>>>>>>>>>>>>>>

- ``load_split_dataset (bool)`` : Whether to load split dataset. Defaults to ``False``.
- ``feature_type (str)`` : Generating Type of node features. Defaults to ``all_one``. Range in ``['all_one', 'all_zero', 'random', 'node_type']``.
- ``task (str)`` : Task name. Defaults to ``satisfiability``. Range in ``['satisfiability', 'maxsat', 'unsat_core']``.
- ``task_type (str)`` : Construction Type of graph. Defaults to ``lcg``.
  If ``model`` in ``['gin', 'gcn', 'gat']``, ``task_type`` in ``['lcg', 'vcg', 'vig', 'lig']``.
  Otherwise ``task_type`` in ``['lcg', 'aig']``.
- ``task_level (str)`` : Task level. Defaults to ``graph``.
  Range in ``['e2e_graph', 'e2e_node', 'e2e_link', 'lr_graph', 'lr_node', 'lr_link', 'graph']``.
- ``dataset_path (str)`` : Path to the dataset. Defaults  ``./dataset/satisfiability``.

- ``model_settings (dict)`` : Settings for the model.

  * ``model (str)`` : Model name. Defaults  ``neurosat``. Range in ``['neurosat', 'neurocore', 'nlocalsat', 'gms', 'deepsat', 'querysat', 'satformer', 'gin', 'gcn', 'gat']``.
  * ``input_size (int)`` : Input dimension of the feature. Defaults to ``1``.
  * ``hidden_size (int)`` : Hidden dimension of the embedding. Defaults to ``128``.
  * ``output_size (int)`` : Output dimension of the embedding. Defaults to ``1``.
  * ``loss (str)`` : Loss function. Defaults to ``binary_cross_entropy``. Range in ``['cross_entropy', 'binary_cross_entropy', 'mse', 'mae']``.
  * ``num_fc (int)`` : Number of fully connected layers. Defaults to ``3``.
  * ``num_round (int)`` : Number of message-passing rounds. Defaults to ``32``.
  * ``dropout_ratio (float)`` : Dropout ratio. Defaults to ``0``.
  * ``sigmoid (bool)`` : Whether to use sigmoid. Defaults to ``True``.
  * ``pooling (str)`` : Pooling type. Defaults to ``mean``. Range in ``['mean', 'max']``.

- ``optimizer (str)`` : Optimizer. Defaults  ``adam``. Range in ``['adam', 'sgd', 'adagrad', 'adadelta', 'rmsprop']``.

- ``scheduler_settings (dict)`` : Settings for the scheduler(Optional).

  * ``scheduler (str)`` : Scheduler name. Defaults to ``ReduceLROnPlateau``. Range in ``['ReduceLROnPlateau', 'StepLR']``.
  * ``patience (int)`` : Patience for the scheduler. Defaults to ``10``.
  * ``factor (float)`` : Factor for the scheduler. Defaults to ``0.5``.
  * ``mode (str)`` : Mode for the scheduler. Defaults to ``min``.

- ``valid_metric (str)`` : Validation metric. Defaults to ``acc``. Range in ``['acc', 'mse', 'mae', 'rmse']``.
- ``eval_metric (str)`` : Evaluation metric. Defaults to ``acc``.
- ``eval_step (int)`` : Evaluation step. Defaults to ``1``.
- ``epochs (int)`` : The number of training epochs. Defaults to ``100``.
- ``early_stop (bool)`` : Whether to early stop. Defaults to ``False``.
- ``lr (float)`` : Learning rate. Defaults to ``1e-4``.
- ``weight_decay (float)`` : Weight decay. Defaults to ``1e-10``.
- ``device (str)`` : Device for training. Defaults to ``cuda:6``.
- ``split_ratio (list)`` : Split ratio for train, validation, test. Defaults to ``[0.6, 0.2, 0.2]``.
- ``batch_size (int)`` : Batch size. Defaults to ``64``.
- ``save_model (str)`` : Path to save the model. Defaults to ``./save_model/neurosat.pt``.
- ``tensorboard_dir (str)`` : Directory for TensorBoard logs. Defaults to ``./tensorboard_run``.
- ``log settings (dict)`` : Log settings.

  * ``log_file (str)`` : Path to the log file. Defaults to ``./log/neurosat.log``.



