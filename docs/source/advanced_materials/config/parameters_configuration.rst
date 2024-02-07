Parameters Configuration
>>>>>>>>>>>>>>>>>>>>>>>>>

SATGL supports three types of parameter configurations: Config files,
Parameter Dicts and Command Line. The parameters are assigned via the
Configuration module.

Config Files
^^^^^^^^^^^^^^^^

Configuration parameters in SATGL are specified using YAML format.Users should structure the config files following YAML
syntax conventions, defining the desired hyperparameters.These YAML configs are then interpreted by SATGL's configuration
handler which validates and loads the parameters to finalize the experiment settings.

To begin with, we write the parameters into the yaml files (e.g. `example.yaml`).

.. code:: yaml

    device: cuda:0
    batch_size: 128

Then, the yaml files are conveyed to the configuration module and convert the default yaml file to finish the
parameter settings.

.. code:: python

    from satgl.config.configurator import Config

    config = Config(config_file_list=['./satgl/yaml/example.yaml'])
    print('device: ', config['device'])
    print('batch_size: ', config['batch_size'])

output:

.. code:: bash

    device: cuda:0
    batch_size: 128

The parameter ``config_file_list`` supports multiple yaml files.

For more details on yaml, please refer to `YAML <https://yaml.org/>`_.

Parameter Dicts
^^^^^^^^^^^^^^^^^^

The parameter configuration is implemented as a Python dictionary where each key represents a parameter name mapped to its
set value. Users can define hyperparameters in a dictionary structure and pass this to the configuration handler for
loading the parameters.

An example is as follows:

.. code:: python

    from satgl.config.configurator import Config

    additional_args = {
         "device": "cuda:0",
         "batch_size": 128,
    }

    config = Config(config_file_list=['./satgl/yaml/example.yaml'], parameter_dict=additional_args)
    print('device: ', config['device'])
    print('batch_size: ', config['batch_size'])

output:

.. code:: bash

    device: cuda:0
    batch_size: 128

Command Line
^^^^^^^^^^^^^^^^^^^^^^^^

We can also assign parameters based on the command line.
The parameters in the command line can be read from the configuration module.
The format is: `--parameter_name=[parameter_value]`.


Write the following code to the python file (e.g. `run.py`):

.. code:: python

    from satgl.config.configurator import Config

    config = Config(config_file_list=['./satgl/yaml/example.yaml'])
    print('device: ', config['device'])
    print('batch_size: ', config['batch_size'])

Running:

.. code:: bash

    python run.py --device=cuda:0 --batch_size=128

output:

.. code:: bash

    device: cuda:0
    batch_size: 128

.. note::

    In SATGL, The priority of the configuration methods is: Command Line > Parameter Dicts > Config Files > Default Settings.

