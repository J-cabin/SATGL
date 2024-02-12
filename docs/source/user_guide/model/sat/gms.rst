GMS
==============

Introduction
------------------

`[paper] <https://api.semanticscholar.org/CorpusID:244117384>`_

**Title:** Can Graph Neural Networks Learn to Solve MaxSAT Problem?

**Authors:** Minghao Liu and Fuqi Jia and Pei Huang and Fan Zhang and Yuchen Sun and Shaowei Cai and Feifei Ma and Jian Zhang

**Abstract:** With the rapid development of deep learning techniques, various recent work has tried to
apply graph neural networks (GNNs) to solve NP-hard problems such as Boolean Satisfiability (SAT), which
shows the potential in bridging the gap between machine learning and symbolic reasoning. However, the
quality of solutions predicted by GNNs has not been well investigated in the literature. In this paper,
we study the capability of GNNs in learning to solve Maximum Satisfiability (MaxSAT) problem, both from
theoretical and practical perspectives. We build two kinds of GNN models to learn the solution of MaxSAT
instances from benchmarks, and show that GNNs have attractive potential to solve MaxSAT problem through
experimental evaluation. We also present a theoretical explanation of the effect that GNNs can learn to
solve MaxSAT problem to some extent for the first time, based on the algorithmic alignment theory.

**Config**

.. code:: python

    load_split_dataset: True
    feature_type: all_one
    task: satisfiability
    task_type: lcg
    task_level: graph
    load_field: ["label:float"]
    dataset_path: ./dataset/my_3_sat_1000

    model_settings:
      model: gms
      input_size: 1
      hidden_size: 128
      output_size: 1
      dropout_ratio: 0
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
    device: cuda:7
    batch_size: 32

    #log settings
    log_file: ./log/old_gms.log
