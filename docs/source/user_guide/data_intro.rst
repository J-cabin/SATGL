.. _data-introduction:

Data
==============

To enhance extensibility and reuse, SATGL's data module implements a processing pipeline to
transform raw data into model-ready inputs.

The workflow functions as follows:

.. image:: ../_static/data_flow.png
    :align: center

The details are as follows:

- Raw Input

    Unprocessed raw input dataset. Detailed as :ref:`data-rawdata`.

- Dataset

    The Dataset is indeed implemented based on ``DGLDataset``. It serves as a foundational structure for creating
    specific datasets related to SAT tasks. The dataset initializes parameters, constructs graphs based on model settings using the
    DGL library, and contains abstract methods to be implemented by subclasses for loading data and defining dataset behavior. This
    design ensures compatibility with DGL and allows for customization when creating datasets for SAT-related tasks.
    Detailed as :ref:`data-dataset`.

- DataLoader

    The DataLoader module provides a convenient interface for loading and batching data from a custom dataset designed
    for SAT tasks. It utilizes DGL's graph representation and extends it to handle various data types efficiently.
    The DataLoader module extends the capabilities of ``DGL`` for efficient handling of datasets during training.
    It inherits from both ``dgl.dataloading.GraphCollator`` and ``dgl.dataloading.GraphDataLoader`` in the ``DGL`` library.
    Detailed as :ref:`data-dataloader`.

Here are the related docs for data module:

.. toctree::
   :maxdepth: 1

   data/raw_input
   data/dataset
   data/dataloader
