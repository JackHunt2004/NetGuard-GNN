import os
import pandas as pd
from .utils import scale_scores

DATA_DIR = "data"

def save_results(u_gnn_norm, ocsvm_user, user_ids, r_gnn, resource_ids):
    ensemble = 0.5 * u_gnn_norm + 0.5 * ocsvm_user
    user_scores = pd.DataFrame({
        "user_id": user_ids,
        "gnn_score": u_gnn_norm,
        "ocsvm_score": ocsvm_user,
        "ensemble_score": ensemble
    }).sort_values("ensemble_score", ascending=False)
    resource_scores = pd.DataFrame({
        "resource_id": resource_ids,
        "gnn_score": scale_scores(r_gnn)
    }).sort_values("gnn_score", ascending=False)
    user_scores.to_csv(os.path.join(DATA_DIR, "user_scores.csv"), index=False)
    resource_scores.to_csv(os.path.join(DATA_DIR, "resource_scores.csv"), index=False)
    return user_scores, resource_scores