import streamlit as st
import pandas as pd
import os

from src.ocsvm import train_ocsvm, get_ocsvm_anomaly_scores
from src.results import save_results
from src.visualization import plot_top_users

DATA_DIR = "data"
TAB_FILE = os.path.join(DATA_DIR, "tabular_features.csv")
CSV_FILE = os.path.join(DATA_DIR, "verilog_output.csv")

st.set_page_config(page_title="NetGuard-GNN", page_icon="🛡️", layout="wide")

st.title("🛡️ NetGuard-GNN – Network Anomaly Detection")
st.write("Upload your **verilog_output.csv** and **tabular_features.csv** to run anomaly detection.")

# File uploaders
tab_file = st.file_uploader("Upload tabular_features.csv", type=["csv"])
csv_file = st.file_uploader("Upload verilog_output.csv", type=["csv"])

if tab_file and csv_file:
    os.makedirs(DATA_DIR, exist_ok=True)
    tab_path = os.path.join(DATA_DIR, "tabular_features.csv")
    csv_path = os.path.join(DATA_DIR, "verilog_output.csv")

    # Save uploaded files
    with open(tab_path, "wb") as f:
        f.write(tab_file.getbuffer())
    with open(csv_path, "wb") as f:
        f.write(csv_file.getbuffer())

    st.success("✅ Files uploaded successfully. Running anomaly detection...")

    # Train OCSVM + get anomaly scores
    ocsvm_model = train_ocsvm(tab_path, DATA_DIR)
    ocsvm_user, user_ids = get_ocsvm_anomaly_scores(ocsvm_model, csv_path, tab_path)

    # Combine into results (GNN skipped for simplicity on Streamlit Cloud)
    dummy_u_gnn = ocsvm_user  # placeholder: use ocsvm scores only
    dummy_r_gnn = [0.0] * len(pd.read_csv(csv_path)["resource_id"].unique())

    resource_ids = pd.read_csv(csv_path)["resource_id"].astype("category").cat.categories.tolist()
    user_scores, resource_scores = save_results(dummy_u_gnn, ocsvm_user, user_ids, dummy_r_gnn, resource_ids, DATA_DIR)

    # Show top suspicious users
    st.write("### Top Suspicious Users")
    st.dataframe(user_scores.head(10))

    # Plot charts
    st.write("### Visualization")
    fig1, fig2 = plot_top_users(user_scores, top_k=5)

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)


    # Download button
    st.download_button(
        label="📥 Download full user scores CSV",
        data=user_scores.to_csv(index=False).encode("utf-8"),
        file_name="user_scores.csv",
        mime="text/csv",
    )

else:
    st.info("👆 Please upload both required files to proceed.")