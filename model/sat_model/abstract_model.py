import dgl
import torch
import numpy as np
import torch.nn as nn

class AbstractSATModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def foward(self, g, x):
        pass

    def predict(self, g):
        pass