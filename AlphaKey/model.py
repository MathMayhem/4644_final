# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation

from config import Config

class PolicyValueNetV4(nn.Module):
    """
    The V4 neural network architecture.
    """
    def __init__(self):
        super(PolicyValueNetV4, self).__init__()

        self.initial_embedding = nn.Linear(Config.NODE_INPUT_DIM, Config.GNN_EMBEDDING_DIM)
        self.positional_embeddings = nn.Embedding(Config.NUM_KEYS, Config.GNN_EMBEDDING_DIM)

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(Config.GAT_LAYERS):
            self.gat_layers.append(
                GATv2Conv(
                    Config.GNN_EMBEDDING_DIM,
                    Config.GNN_EMBEDDING_DIM // Config.NUM_ATTENTION_HEADS,
                    heads=Config.NUM_ATTENTION_HEADS
                )
            )
            self.norm_layers.append(nn.LayerNorm(Config.GNN_EMBEDDING_DIM))

        gate_nn = nn.Sequential(
            nn.Linear(Config.GNN_EMBEDDING_DIM, Config.ATTENTION_GATE_DIM),
            nn.ReLU(),
            nn.Linear(Config.ATTENTION_GATE_DIM, 1)
        )
        self.attention_pooling = AttentionalAggregation(gate_nn)

        self.policy_head = nn.Sequential(
            nn.Linear(Config.FINAL_EMBEDDING_DIM, Config.HEAD_HIDDEN_DIM),
            nn.LayerNorm(Config.HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HEAD_HIDDEN_DIM, Config.NUM_ACTIONS)
        )
        self.value_head = nn.Sequential(
            nn.Linear(Config.FINAL_EMBEDDING_DIM, Config.HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HEAD_HIDDEN_DIM, Config.NUM_WEIGHTS),
            nn.Tanh()
        )

    def forward(self, batch):
        x, edge_index, batch_map = batch.x, batch.edge_index, batch.batch

        x = F.relu(self.initial_embedding(x))

        positions = torch.arange(Config.NUM_KEYS, device=x.device).repeat(batch.num_graphs)
        pos_enc = self.positional_embeddings(positions)
        x = x + pos_enc

        for i, layer in enumerate(self.gat_layers):
            x = x + F.relu(layer(x, edge_index))
            x = self.norm_layers[i](x)

        graph_embedding = self.attention_pooling(x, batch_map)

        policy_logits = self.policy_head(graph_embedding)
        value_vector = self.value_head(graph_embedding)

        return policy_logits, value_vector
