Quick Start
==========================

For general recommendation, we choose **NeuroSAT** model to show how to train, valid and test it on **satisfiability task**
from both **API** and **source code**.

Quick Start From API
---------------------

1. Prepare your data:
>>>>>>>>>>>>>>>>>>>>>>

Firstly, you need to prepare and load data before running a model. To help users quickly get start, SATGL has some build-in cnf dataset
and you can directly use it. However, if you want to use other datasets, you can read **refref** for more information.

Then, you need to set data config for data loading. You can create a yaml file called neurosat.yaml and write the following settings:

.. code-block:: python

    # dataset config : General Recommendation
    dataset_name: neurosat
    load_split_dataset: True
    feature_type: all_one
    dataset_path: ./dataset/satisfiability

2. Choose a model
>>>>>>>>>>>>>>>>>>>>

You can choose a model from our **model** and set the parameter for the model. Here we choose NeuroSAT model and add
settings into the neurosat.yaml, like:

.. code-block:: python

    # model config
    model_settings:
      model: neurosat
      input_size: 1
      hidden_size: 128
      output_size: 1
      loss: binary_cross_entropy
      num_fc: 3
      num_round: 32
      dropout_ratio: 0
      sigmoid: True
      pooling: mean

3. Set training and evaluation config:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
SATGL provides multiple training and evaluation approaches. You can select the training and testing methodology for
your model simply by configuring the settings.

In this case, our goal is to train and evaluate the NeuroSAT model utilizing the training-validation-test approach. This involves
optimizing the model parameters using the training data, selecting the optimal parameters based on validation set performance,
and finalizing the model testing on the unseen test set.

To enable this methodology and evaluate the full ranking performance over all candidate items, we can add the following
customizations to the neurosat.yaml configuration file:

.. code-block:: python

    # train settings
    task: satisfiability
    task_type: lcg
    task_level: graph

    scheduler_settings:
      scheduler: ReduceLROnPlateau
      patience: 10
      factor: 0.5
      mode: min

    valid_metric: acc
    epochs: 100
    lr: 1e-4
    weight_decay: 1e-10
    device: cuda:6
    split_ratio: [0.6, 0.2, 0.2]
    batch_size: 64
    save_model: ./save_model/neurosat.pt
    scheduler: ReduceLROnPlateau


    #log settings
    log_file: ./log/neurosat.log

For more details of training and evaluation config, please refer to **config**.

4. Run the model and collect the result:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

You can create a new python file (e.g., neurosat_test.py), and write the following code:

.. code-block:: python

    import torch
    import dgl
    from satgl.config.configurator import Config
    from satgl.trainer.trainer import AbstractTrainer
    from satgl.utils.utils import seed_everything
    from satgl.data.utils import get_dataset, get_dataloader
    from satgl.model.sat_model.neurosat import NeuroSAT
    from satgl.metric.metric import eval_accuracy
    from satgl.trainer.trainer import TaskTrainer
    from satgl.data.utils import get_data
    from satgl.model.utils import get_model
    from satgl.data.wrapper import SATDataWrapper

    def run_experiment(additional_args=None):
    seed_everything(0, reproducibility=True)
    config = Config(config_file_list=['./satgl/yaml/neurosat.yaml'], parameter_dict=additional_args)
    sat_data = SATDataWrapper(config)
    model = get_model(config=config)
    Trainer = TaskTrainer(config=config, model=model)
    Trainer.train(sat_data.train_dataloader, valid_loader=sat_data.valid_dataloader, test_loader=sat_data.test_dataloader)

    run_experiment()

Then run the following command:

.. code-block:: python

    python neurosat_test.py

And you will obtain the output like:

.. code::

    processing train dataset ...
    processing valid dataset ...
    processing test dataset ...
    epoch [0/100]
    train | acc : 0.470000 | data_size : 1600.000000 | loss : 0.693869
    valid | acc : 0.500000 | data_size : 200.000000 | loss : 0.692978
    epoch [1/100]
    train | acc : 0.521875 | data_size : 1600.000000 | loss : 0.693096
    valid | acc : 0.500000 | data_size : 200.000000 | loss : 0.693015
    epoch [2/100]
    train | acc : 0.497500 | data_size : 1600.000000 | loss : 0.693087
    valid | acc : 0.500000 | data_size : 200.000000 | loss : 0.692935
    epoch [3/100]
    train | acc : 0.496875 | data_size : 1600.000000 | loss : 0.693105
    valid | acc : 0.515000 | data_size : 200.000000 | loss : 0.692951
    epoch [4/100]
    train | acc : 0.492500 | data_size : 1600.000000 | loss : 0.693318
    valid | acc : 0.500000 | data_size : 200.000000 | loss : 0.692879

The above is the whole process of running a model in SATGL, and you can read other docs for depth usage.

Quick Start From Source
------------------------

In addition to the API, you can also directly execute SATGL's source code. The overall process resembles that of the API quick start guide.

To begin, you can create a YAML configuration file named neurosat.yaml. Within this file, specify all model and evaluation parameters as follows:

.. code-block:: python

    # neurosat.yaml
    dataset_name: neurosat
    load_split_dataset: True
    feature_type: all_one
    task: satisfiability
    task_type: lcg
    task_level: graph
    dataset_path: ./dataset/satisfiability

    model_settings:
      model: neurosat
      input_size: 1
      hidden_size: 128
      output_size: 1
      loss: binary_cross_entropy
      num_fc: 3
      num_round: 32
      dropout_ratio: 0
      sigmoid: True
      pooling: mean

    scheduler_settings:
      scheduler: ReduceLROnPlateau
      patience: 10
      factor: 0.5
      mode: min

    # train settings
    valid_metric: acc
    epochs: 100
    lr: 1e-4
    weight_decay: 1e-10
    device: cuda:6
    split_ratio: [0.6, 0.2, 0.2]
    batch_size: 64
    save_model: ./save_model/neurosat.pt
    scheduler: ReduceLROnPlateau

    #log settings
    log_file: ./log/neurosat.log

Then run the following command:

.. code-block:: python

    python neurosat_test.py

If you want to change the task, dataset_path or other hyper-parameters, you can run the following command like:

.. code-block:: python

    python neurosat_test.py --task maxsat --dataset_path ./dataset/maxsat --batch_size 2 --epochs 10

