import os
import pickle
import pandas as pd
from src.data_loader import load_graph
from src.utils import split_edges, scale_scores
from src.train import train_gnn
from src.anomaly import get_gnn_anomaly_scores
from src.ocsvm import train_ocsvm, get_ocsvm_anomaly_scores
from src.results import save_results

# Always resolve data folder relative to this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CSV_FILE = os.path.join(DATA_DIR, "verilog_output.csv")

def main():
    # Pass the absolute path to load_graph so it can find graph_data.pt
    data = load_graph(os.path.join(DATA_DIR, "graph_data.pt"))

    train_pos, _ = split_edges(data[('user', 'accessed', 'resource')].edge_index)

    gnn = train_gnn(data, train_pos)
    weights = {name: param.detach().cpu() for name, param in gnn.named_parameters()}
    with open(os.path.join(DATA_DIR, "gnn_model_manual.pth"), "wb") as f:
        pickle.dump(weights, f)

    u_gnn, r_gnn = get_gnn_anomaly_scores(gnn, data)
    u_gnn_norm = scale_scores(u_gnn)

    ocsvm_model = train_ocsvm()
    ocsvm_user, user_ids = get_ocsvm_anomaly_scores(ocsvm_model)

    resource_ids = pd.read_csv(CSV_FILE)["resource_id"].astype("category").cat.categories.tolist()
    user_scores, _ = save_results(u_gnn_norm, ocsvm_user, user_ids, r_gnn, resource_ids)

    print("\nTop suspicious users:")
    print(user_scores.head(min(5, len(user_scores))))

if __name__ == "__main__":
    main()