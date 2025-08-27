import os, pickle, torch, pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage
from src.utils import split_edges, scale_scores
from src.gnn_model import train_gnn, get_gnn_anomaly_scores
from src.ocsvm import train_ocsvm, get_ocsvm_anomaly_scores
from src.results import save_results
from src.visualization import plot_top_users

torch.serialization.add_safe_globals([BaseStorage])

DATA_DIR = "data"
GRAPH_FILE = os.path.join(DATA_DIR, "graph_data.pt")
TAB_FILE = os.path.join(DATA_DIR, "tabular_features.csv")
CSV_FILE = os.path.join(DATA_DIR, "verilog_output.csv")

def load_graph():
    data: HeteroData = torch.load(GRAPH_FILE, weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ntype in data.node_types:
        if hasattr(data[ntype], "x") and data[ntype].x is not None:
            x = data[ntype].x
            x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
            data[ntype].x = x

    return data.to(device)

def main():
    data = load_graph()
    train_pos, _ = split_edges(data[('user', 'accessed', 'resource')].edge_index)

    gnn = train_gnn(data, train_pos)
    weights = {name: param.detach().cpu() for name, param in gnn.named_parameters()}
    with open(os.path.join(DATA_DIR, "gnn_model_manual.pth"), "wb") as f:
        pickle.dump(weights, f)

    u_gnn, r_gnn = get_gnn_anomaly_scores(gnn, data)
    u_gnn_norm = scale_scores(u_gnn)

    ocsvm_model = train_ocsvm(TAB_FILE, DATA_DIR)
    ocsvm_user, user_ids = get_ocsvm_anomaly_scores(ocsvm_model, CSV_FILE, TAB_FILE)

    resource_ids = pd.read_csv(CSV_FILE)["resource_id"].astype("category").cat.categories.tolist()
    user_scores, _ = save_results(u_gnn_norm, ocsvm_user, user_ids, r_gnn, resource_ids, DATA_DIR)

    print("\nTop suspicious users:")
    print(user_scores.head(min(5, len(user_scores))))

    plot_top_users(user_scores, top_k=5)

if __name__ == "__main__":
    main()