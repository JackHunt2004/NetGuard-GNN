import os
import torch
from torch_geometric.data import HeteroData

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
GRAPH_FILE = os.path.join(DATA_DIR, "graph_data.pt")

def load_graph(path=GRAPH_FILE):
    """Load the hetero graph from the given path."""
    data: HeteroData = torch.load(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return data.to(device)