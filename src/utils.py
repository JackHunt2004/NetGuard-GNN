import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch

def scale_scores(x: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_2d = x.reshape(-1, 1)
    scaler.fit(x_2d)
    return scaler.transform(x_2d).ravel()

def split_edges(edge_index, train_ratio=0.85):
    edges = edge_index.cpu().numpy().T
    train_edges, val_edges = train_test_split(edges, train_size=train_ratio, shuffle=True, random_state=42)
    train_edges = torch.tensor(train_edges.T, dtype=torch.long, device=edge_index.device)
    val_edges = torch.tensor(val_edges.T, dtype=torch.long, device=edge_index.device)
    return train_edges, val_edges