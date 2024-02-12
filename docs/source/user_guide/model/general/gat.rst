GAT
==============

Introduction
------------------

`[paper] <https://api.semanticscholar.org/CorpusID:3292002>`_

**Title:** Graph Attention Networks

**Authors:** Petar Velickovic and Guillem Cucurull and Arantxa Casanova and Adriana Romero and Pietro Lio' and Yoshua Bengio

**Abstract:** We present graph attention networks (GATs), novel neural network architectures that operate
on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior
methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to
attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different
nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or
depending on knowing the graph structure upfront. In this way, we address several key challenges of
spectral-based graph neural networks simultaneously, and make our model readily applicable to inductive
as well as transductive problems. Our GAT models have achieved or matched state-of-the-art results across
four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation
network datasets, as well as a protein-protein interaction dataset (wherein test graphs remain unseen
during training).

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
      model: gat
      num_layers: 32
      num_heads: 8
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
    batch_size: 64

    #log settings
    log_file: ./log/gcn.log



