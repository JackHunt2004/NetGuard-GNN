import torch
import torch.nn as nn
import torch
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.utils import negative_sampling
from .utils import scale_scores

class HeteroGNN(nn.Module):
    def __init__(self, hidden=64, dropout=0.3):
        super().__init__()
        self.convs = HeteroConv({
            ('user', 'accessed', 'resource'): SAGEConv((-1, -1), hidden),
            ('resource', 'rev_accessed', 'user'): SAGEConv((-1, -1), hidden)
        }, aggr='mean')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.convs(x_dict, edge_index_dict)
        x_dict = {k: self.dropout(v.relu()) for k, v in x_dict.items()}
        return x_dict

def edge_score(u_emb, r_emb):
    return (u_emb * r_emb).sum(dim=-1)

def train_gnn(data, train_pos, epochs=50):
    device = data['user'].x.device
    gnn = HeteroGNN(hidden=64).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(gnn.parameters(), lr=5e-4, weight_decay=1e-4)

    num_users = data['user'].x.size(0)
    num_resources = data['resource'].x.size(0)

    for epoch in range(epochs):
        gnn.train()
        opt.zero_grad()

        out = gnn(
            {'user': data['user'].x, 'resource': data['resource'].x},
            {
                ('user', 'accessed', 'resource'): train_pos,
                ('resource', 'rev_accessed', 'user'): train_pos.flip(0)
            }
        )

        user_z, res_z = out['user'], out['resource']
        u_pos, r_pos = train_pos[0], train_pos[1]

        pos_logits = edge_score(user_z[u_pos], res_z[r_pos])

        # Negative sampling
        edge_index_shifted = torch.stack([train_pos[0], train_pos[1] + num_users])
        neg_edges_shifted = negative_sampling(
            edge_index=edge_index_shifted,
            num_nodes=num_users + num_resources,
            num_neg_samples=u_pos.size(0),
            method='sparse'
        )
        u_neg = neg_edges_shifted[0]
        r_neg = neg_edges_shifted[1] - num_users
        mask = (u_neg < num_users) & (r_neg >= 0) & (r_neg < num_resources)
        u_neg, r_neg = u_neg[mask], r_neg[mask]
        neg_logits = edge_score(user_z[u_neg], res_z[r_neg])

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)

        loss = bce(logits, labels)
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    return gnn

def get_gnn_anomaly_scores(gnn, data):
    device = next(gnn.parameters()).device
    gnn.eval()

    edge_index_dict = {
        ('user', 'accessed', 'resource'): data[('user', 'accessed', 'resource')].edge_index.to(device),
        ('resource', 'rev_accessed', 'user'): data[('user', 'accessed', 'resource')].edge_index.flip(0).to(device)
    }
    x_dict = {
        'user': data['user'].x.to(device),
        'resource': data['resource'].x.to(device)
    }

    with torch.no_grad():
        emb = gnn(x_dict, edge_index_dict)

    user_z, res_z = emb['user'], emb['resource']
    u_all, r_all = edge_index_dict[('user', 'accessed', 'resource')]

    lp = torch.sigmoid(edge_score(user_z[u_all], res_z[r_all]))

    u_scores = torch.zeros(user_z.size(0), device=device)
    u_count = torch.zeros_like(u_scores)
    r_scores = torch.zeros(res_z.size(0), device=device)
    r_count = torch.zeros_like(r_scores)

    for i in range(u_all.size(0)):
        u, r = u_all[i], r_all[i]
        surprisal = 1.0 - lp[i]
        u_scores[u] += surprisal
        u_count[u] += 1
        r_scores[r] += surprisal
        r_count[r] += 1

    u_scores = scale_scores((u_scores / torch.clamp(u_count, min=1)).cpu().numpy())
    r_scores = scale_scores((r_scores / torch.clamp(r_count, min=1)).cpu().numpy())

    return u_scores, r_scores