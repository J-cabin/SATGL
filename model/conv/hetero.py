import dgl.function as fn
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, Parameter

class HeteroConv(nn.Module):
    """
        HeteroConv pass the src_node's feature to dst_node
    """
    def __init__(self):
        super(HeteroConv, self).__init__()

    def forward(self, g, src_type, e_type, dst_type, src_embedding):
        rel_g = g[src_type, e_type, dst_type]
        with rel_g.local_scope():
            rel_g.nodes[src_type].data["h"] = src_embedding
            rel_g.apply_edges(fn.copy_u("h", "m"))
            rel_g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))
            dst_embedding = rel_g.nodes[dst_type].data["h"]
        return dst_embedding

class HeteroGCNConv(nn.Module):
    """
        HeteroGCNConv passes the src_node's feature to dst_node with normalization.
    """
    def __init__(self):
        super(HeteroGCNConv, self).__init__()

    def forward(self, g, src_type, e_type, dst_type, src_embedding):
        rel_g = g[src_type, e_type, dst_type]
        with rel_g.local_scope():
            rel_g.nodes[src_type].data["h"] = src_embedding

            # Compute degree matrix and its inverse square root
            src_deg_inv_sqrt = torch.pow(rel_g.out_degrees().float().clamp(min=1), -0.5)
            dst_deg_inv_sqrt = torch.pow(rel_g.in_degrees().float().clamp(min=1), -0.5)
            
            # message passing
            rel_g.nodes[src_type].data["h"] = src_embedding * src_deg_inv_sqrt.unsqueeze(-1)
            rel_g.apply_edges(fn.copy_u("h", "m"))
            rel_g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))
            dst_embedding = rel_g.nodes[dst_type].data["h"] * dst_deg_inv_sqrt.unsqueeze(-1)
        
        return dst_embedding