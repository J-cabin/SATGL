import yaml
import os
import re
import sys

"""
    todo:
        -load dict from cmd
        -replace print to logger
"""

class Config(object):
    """
        SAT config
    """

    def __init__(
        self, config_file_list=None, parameter_dict=None
    ):
        # fixed parameter
        cur_path = os.path.dirname(os.path.abspath(__file__))   
        self.default_config_file_list = [os.path.join(cur_path, "../yaml/default_config.yaml")]

        # copy parameter
        self.config_file_list = config_file_list
        self.parameter_dict = parameter_dict
        
        # build parameter
        self.default_dict = self._load_default_dict()
        self.file_dict = self._load_file_dict(config_file_list)
        self.parameter_dict = self._load_parameter_dict()
        self.cmd_dict = self._load_cmd_dict()
        self.config_dict = self._merge_dict()      
    
    def _load_parameter_dict(self):
        if self.parameter_dict:
            return self.parameter_dict
        return dict()

    def _load_cmd_dict(self):
        cmd_dict = dict()
        execution_environments = set(
            ["ipykernel_launcher", "colab"]
        )
        unrecognized_args = []

        if sys.argv[0] not in execution_environments:
            for arg in sys.argv[1:]:
                if arg.startswith('--') and len(arg.split('=')) == 2:
                    arg_key, arg_value = arg.split('=')
                    cmd_dict[arg_key[2:]] = arg_value
                else:
                    unrecognized_args.append(arg)
        if len(unrecognized_args) > 0:
            print("args [{}] be ignored".format(" | ".join(unrecognized_args)))
        return cmd_dict

    def _load_file_dict(self, config_file_list):
        file_dict = dict()
        if config_file_list:
            for file in config_file_list:
                with open(file, "r", encoding="utf-8") as f:
                    file_dict.update(
                        yaml.load(f.read(), Loader=yaml.FullLoader)
                    )
        return file_dict

    def _load_default_dict(self):
        return self._load_file_dict(self.default_config_file_list)

    def _merge_dict(self):
        config_dict = self.default_dict
        config_dict.update(self.file_dict)
        config_dict.update(self.parameter_dict)
        config_dict.update(self.cmd_dict)
        return config_dict
        
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("key must be str")
        self.config_dict[key] = value
    
    def __getattr__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            raise AttributeError("No such attribute: {}".format(item))
    
    def __getitem__(self, item):
        if item in self.config_dict:
            return self.config_dict[item]
        else:
            raise KeyError("No such key: {}".format(item))

    def __str__(self):
        return str(self.config_dict)
    
    def __repr__(self):
        return self.__str__()