import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSATConv(nn.Module):
    def __init__(self, emb_size):
        super(DeepSATConv, self).__init__()
        self.neighbour_fc = nn.Linear(emb_size, emb_size)
        self.self_fc = nn.Linear(emb_size, emb_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.neighbour_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.self_fc.weight, gain=gain)

    def message_func(self, edges):
        return {
            "h": edges.src["h"],
            "msg": edges.src["self_h"] + edges.dst["neibour_h"]
        }
    
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['msg'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, graph, h):
        # linear transformation
        neibour_h = self.neighbour_fc(h)
        self_h = self.self_fc(h)

        # message passing
        graph.ndata["h"] = h
        graph.ndata["neibour_h"] = neibour_h
        graph.ndata["self_h"] = self_h
        graph.update_all(self.message_func, self.reduce_func)

        # pop
        h = graph.ndata.pop("h")
        graph.ndata.pop("neibour_h")
        graph.ndata.pop("self_h")

        return h