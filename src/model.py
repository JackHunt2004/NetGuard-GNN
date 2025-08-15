import torch.nn as nn
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.convs = HeteroConv({
            ('user', 'accessed', 'resource'): SAGEConv((-1, -1), hidden),
            ('resource', 'rev_accessed', 'user'): SAGEConv((-1, -1), hidden)
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.convs(x_dict, edge_index_dict)
        return {k: v.relu() for k, v in x_dict.items()}

def edge_score(u_emb, r_emb):
    return (u_emb * r_emb).sum(dim=-1)