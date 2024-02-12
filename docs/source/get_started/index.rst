Install
=================

SATGL works with the following operating systems:

* Ubuntu 18.04

Python environment requirments

- `Python <https://www.python.org/>`_ >= 3.9
- `PyTorch <https://pytorch.org/>`_  >= 2.0.1
- `DGL <https://github.com/dmlc/dgl>`_ >= 1.1.1

**1. Python environment (Optional):** We recommend using Conda package manager

.. code:: bash

    conda create -n satgl python=3.9
    source activate satgl

**2. Pytorch:** Follow their `tutorial <https://pytorch.org/get-started/>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install torch torchvision torchaudio

**3. DGL:** Follow their `tutorial <https://www.dgl.ai/pages/start.html>`_ to run the proper command according to
your OS and CUDA version. For example:

.. code:: bash

    pip install dgl -f https://data.dgl.ai/wheels/repo.html

**4. Install satgl:**

* install from source

.. code:: python

    git clone https://github.com/BUPT-GAMMA/SATGL
    # If you encounter a network error, try git clone from openi as following.
    # git clone https://git.openi.org.cn/GAMMALab/SATGL.git
    cd SATGL
    pip install -r requirements.txt
