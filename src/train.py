import torch
from torch import nn
from torch_geometric.utils import negative_sampling
from .model import HeteroGNN, edge_score

def train_gnn(data, train_pos, epochs=50):
    device = data['user'].x.device
    gnn = HeteroGNN(hidden=64).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(epochs):
        gnn.train()
        opt.zero_grad()
        out = gnn({'user': data['user'].x, 'resource': data['resource'].x}, {
            ('user', 'accessed', 'resource'): train_pos,
            ('resource', 'rev_accessed', 'user'): train_pos.flip(0)
        })

        user_z, res_z = out['user'], out['resource']
        u_pos, r_pos = train_pos[0], train_pos[1]
        pos_logits = edge_score(user_z[u_pos], res_z[r_pos])

        neg_edges = negative_sampling(
            edge_index=train_pos,
            num_nodes=(data['user'].x.size(0), data['resource'].x.size(0)),
            num_neg_samples=u_pos.size(0),
            method='sparse'
        )
        u_neg, r_neg = neg_edges[0], neg_edges[1]
        neg_logits = edge_score(user_z[u_neg], res_z[r_neg])

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        loss = bce(logits, labels)
        loss.backward()
        opt.step()

    return gnn