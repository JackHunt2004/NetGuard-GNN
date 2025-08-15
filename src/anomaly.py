import torch
from .model import edge_score

def get_gnn_anomaly_scores(gnn, data):
    device = next(gnn.parameters()).device
    gnn.eval()

    x_dict = {'user': data['user'].x.to(device), 'resource': data['resource'].x.to(device)}
    edge_index_dict = {
        ('user', 'accessed', 'resource'): data[('user', 'accessed', 'resource')].edge_index.to(device),
        ('resource', 'rev_accessed', 'user'): data[('user', 'accessed', 'resource')].edge_index.flip(0).to(device)
    }

    with torch.no_grad():
        emb = gnn(x_dict, edge_index_dict)

    u_all = edge_index_dict[('user', 'accessed', 'resource')][0]
    r_all = edge_index_dict[('user', 'accessed', 'resource')][1]
    lp = torch.sigmoid(edge_score(emb['user'][u_all], emb['resource'][r_all]))

    u_scores = torch.zeros(emb['user'].size(0), device=device)
    u_count = torch.zeros_like(u_scores)
    r_scores = torch.zeros(emb['resource'].size(0), device=device)
    r_count = torch.zeros_like(r_scores)

    for i in range(u_all.size(0)):
        u, r = u_all[i], r_all[i]
        surprisal = 1.0 - lp[i]
        u_scores[u] += surprisal
        u_count[u] += 1
        r_scores[r] += surprisal
        r_count[r] += 1

    u_scores = (u_scores / torch.clamp(u_count, min=1)).cpu().numpy()
    r_scores = (r_scores / torch.clamp(r_count, min=1)).cpu().numpy()

    return u_scores, r_scores