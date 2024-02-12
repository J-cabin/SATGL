NeuroSAT
==============

Introduction
------------------

`[paper] <https://api.semanticscholar.org/CorpusID:3632319>`_

**Title:** Learning a SAT Solver from Single-Bit Supervision

**Authors:** Daniel Selsam and Matthew Lamm and Benedikt B{\"u}nz and Percy Liang and Leonardo Mendon?a de Moura and David L. Dill

**Abstract:** We present NeuroSAT, a message passing neural network that learns to solve SAT problems after
only being trained as a classifier to predict satisfiability. Although it is not competitive with
state-of-the-art SAT solvers, NeuroSAT can solve problems that are substantially larger and more
difficult than it ever saw during training by simply running for more iterations. Moreover,
NeuroSAT generalizes to novel distributions; after training only on random SAT problems, at test time
it can solve SAT problems encoding graph coloring, clique detection, dominating set, and vertex cover
problems, all on a range of distributions over small random graphs.

**Config**

.. code:: python

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

