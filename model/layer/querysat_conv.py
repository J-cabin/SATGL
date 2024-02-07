import torch
import torch.nn as nn
import torch.nn.functional as F


class QuerySATConv(nn.Module):
    def __init__(self):
        super(QuerySATConv, self).__init__()

    def message_func(self, edges):
        # dst node h as message
        return {
            "msg": edges.dst["h"]
        }
    
    def reduce_func(self, nodes):
        # prod all dst embedding
        h = torch.prod(nodes.mailbox['msg'],dim=1)
        return {'h': h}

    def forward(self, graph, h):
        # message passing
        graph.ndata["h"] = h
        graph.update_all(self.message_func, self.reduce_func)

        # pop h from graph
        h = graph.ndata.pop("h")

        return h