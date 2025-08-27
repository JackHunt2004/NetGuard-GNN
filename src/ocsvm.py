import os
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump
from .utils import scale_scores

def train_ocsvm(tab_file, data_dir):
    tab = pd.read_csv(tab_file)
    feat_cols = tab.columns.tolist()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ocsvm", OneClassSVM(kernel="rbf", gamma="scale", nu=0.05))
    ])
    pipe.fit(tab[feat_cols])

    dump(pipe, os.path.join(data_dir, "ocsvm.joblib"))
    return pipe

def get_ocsvm_anomaly_scores(pipe, csv_file, tab_file):
    raw = pd.read_csv(csv_file, parse_dates=["timestamp"])
    tab = pd.read_csv(tab_file)

    dfn = pipe.decision_function(tab)
    ocsvm_event_anom = scale_scores(-dfn)
    raw["ocsvm_event_anom"] = ocsvm_event_anom

    ocsvm_user = raw.groupby("user_id")["ocsvm_event_anom"].mean().reindex(
        raw["user_id"].astype("category").cat.categories
    ).fillna(0.0).values
    ocsvm_user = scale_scores(ocsvm_user)
    user_ids = raw["user_id"].astype("category").cat.categories.tolist()

    return ocsvm_user, user_ids