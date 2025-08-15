# NetGuard-GNN

NetGuard-GNN is a Graph Neural Network (GNN)-based anomaly detection system designed to detect suspicious user and resource activity in network data.  
It combines a **Heterogeneous GNN** with **One-Class SVM (OCSVM)** to produce an ensemble anomaly score.

---

## Features
- Loads heterogeneous graph data (user ↔ resource interactions).
- Trains a GNN using GraphSAGE layers via PyTorch Geometric.
- Extracts anomaly scores from learned node embeddings.
- Uses One-Class SVM for tabular feature anomaly detection.
- Combines both methods into a weighted ensemble score.
- Saves ranked suspicious users and resources to CSV.

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/JackHunt2004/NetGuard-GNN.git
cd NetGuard-GNN