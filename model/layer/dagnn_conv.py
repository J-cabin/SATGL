import argparse

import dgl.function as fn

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Parameter

class DAGNNConv(nn.Module):
    def __init__(self, in_dim, k):
        super(DAGNNConv, self).__init__()

        self.s = Parameter(torch.FloatTensor(in_dim, 1))
        self.k = k

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):
        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata["h"]
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = F.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H