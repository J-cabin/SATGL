GCN
==============

Introduction
------------------

`[paper] <https://api.semanticscholar.org/CorpusID:3144218>`_

**Title:** Semi-Supervised Classification with Graph Convolutional Networks

**Authors:** Thomas Kipf and Max Welling

**Abstract:** We present a scalable approach for semi-supervised learning on graph-structured data that
is based on an efficient variant of convolutional neural networks which operate directly on graphs.
We motivate the choice of our convolutional architecture via a localized first-order approximation of
spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden
layer representations that encode both local graph structure and features of nodes. In a number of
experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach
outperforms related methods by a significant margin.

**Config**

.. code:: python

    dataset_name: neurosat
    load_split_dataset: True
    feature_type: all_one
    task: maxsat
    task_type: lcg
    task_level: graph
    dataset_path: ./dataset/maxsat

    model_settings:
      model: gcn
      num_layers: 32
      hidden_size: 128
      dropout_ratio: 0
      loss: binary_cross_entropy
      num_fc: 3
      sigmoid: True

    scheduler_settings:
      scheduler: ReduceLROnPlateau
      patience: 10
      factor: 0.5
      mode: min

    # train settings
    valid_metric: acc
    epochs: 100
    lr: 0.0001
    weight_decay: 1e-10
    device: cuda:7
    batch_size: 4

    #log settings
    log_file: ./log/gcn.log

