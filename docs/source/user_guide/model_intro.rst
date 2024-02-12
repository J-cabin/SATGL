.. _model-introduction:

Model
==============

Model Usage
----------------

A model serves as an encoder, ingesting graphs along with node features.
It outputs a dictionary of node embeddings, generally encoding all nodes.
Target nodes involved in loss computation may also be exclusively embedded.

The model is composed of two key components, builder and propagator. The builder constructs the model architecture
while the propagator passes input graphs through to generate the node encoding output.

An example can be found in the following code:

.. code:: python

    import torch
    import dgl
    import torch.nn as nn
    from torch.nn.functional import softmax
    from satgl.model.layer.mlp import MLP
    from satgl.model.conv.hetero import HeteroConv


    class NeuroSAT(nn.Module):
        """
            Recurrent Graph Neural Networks of NeuroSAT.
        """
        def __init__(self, config):
            super(NeuroSAT, self).__init__()
            self.config = config

            # check config
            if config["graph_type"] not in ["lcg"]:
                raise ValueError("NeuroSAT only support lcg graph.")

            self.device = config.device
            self.hidden_size = config.model_settings["hidden_size"]
            self.output_size = config.model_settings["output_size"]
            self.num_fc = config.model_settings["num_fc"]
            self.num_round = config.model_settings["num_round"]

            self.l_msg_mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
            self.c_msg_mlp = MLP(self.hidden_size, self.hidden_size, self.hidden_size, num_layer=self.num_fc)
            self.l_update = nn.LSTM(self.hidden_size * 2, self.hidden_size)
            self.c_update = nn.LSTM(self.hidden_size, self.hidden_size)

            self.conv = HeteroConv()

        def forward(self, lcg_graph, l_embedding, c_embedding):
            num_literal = l_embedding.shape[0]
            num_clause = c_embedding.shape[0]

            l_state = (l_embedding.reshape(1, num_literal, -1), torch.zeros(1, num_literal, self.hidden_size).to(self.device))
            c_state = (c_embedding.reshape(1, num_clause, -1), torch.zeros(1, num_clause, self.hidden_size).to(self.device))

            for round_idx in enumerate(range(self.num_round)):
                # literal message passing
                l_hidden = l_state[0].squeeze(0)
                l_msg = self.l_msg_mlp(l_hidden)
                pos_l_msg, neg_l_msg = torch.chunk(l_msg, 2, dim=0)
                pos_l2c_msg = self.conv(lcg_graph, "pos_l", "pos_l2c", "c", pos_l_msg)
                neg_l2c_msg = self.conv(lcg_graph, "neg_l", "neg_l2c", "c", neg_l_msg)
                l2c_msg = pos_l2c_msg + neg_l2c_msg


                # clause message passing
                c_hidden = c_state[0].squeeze(0)
                c_msg = self.c_msg_mlp(c_hidden)
                pos_c2l_msg = self.conv(lcg_graph, "c", "pos_c2l", "pos_l", c_msg)
                neg_c2l_msg = self.conv(lcg_graph, "c", "neg_c2l", "neg_l", c_msg)
                c2l_msg = torch.cat([pos_c2l_msg, neg_c2l_msg], dim=0)
                pos_l_hidden, neg_l_hidden = torch.chunk(l_hidden, 2, dim=0)
                flip_l_hidden = torch.cat([neg_l_hidden, pos_l_hidden], dim=0)

                # update
                _, c_state = self.c_update(l2c_msg.unsqueeze(0), c_state)
                _, l_state = self.l_update(torch.cat([c2l_msg, flip_l_hidden], dim=1).unsqueeze(0), l_state)

            l_final_embedding = l_state[0].squeeze(0)
            c_finla_embedding = c_state[0].squeeze(0)
            return l_final_embedding, c_finla_embedding

Output: All node embeddings or just the target node embedding which is involved in loss calculation.


Model Introduction
-----------------------

We implement 10 models covering sat model and general model. A brief introduction to these models are
as follows:

SAT Model
>>>>>>>>>>>>>>>>>>>>>>>>

.. toctree::
   :maxdepth: 1

   model/sat/deepsat
   model/sat/gms
   model/sat/neurocore
   model/sat/neurosat
   model/sat/nlocalsat
   model/sat/querysat
   model/sat/satformer

General Model
>>>>>>>>>>>>>>>>>>>>>

.. toctree::
   :maxdepth: 1

   model/general/gin
   model/general/gcn
   model/general/gat




