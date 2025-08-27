import plotly.express as px

def plot_top_users(user_scores, top_k=5):
    top_users = user_scores.head(top_k)

    # --- Bar Chart ---
    fig_bar = px.bar(
        top_users,
        x="user_id",
        y="ensemble_score",
        text="ensemble_score",
        labels={"user_id": "User ID", "ensemble_score": "Anomaly Score"},
        title=f"Top {top_k} Suspicious Users (Bar Chart)",
        width=500,   # ✅ shrink figure size
        height=350
    )
    fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=12)
    )

    # --- Pie Chart ---
    fig_pie = px.pie(
        top_users,
        names="user_id",
        values="ensemble_score",
        title=f"Top {top_k} Suspicious Users (Pie Chart)",
        width=400,   # ✅ smaller pie
        height=350
    )
    fig_pie.update_traces(textinfo="percent+label", pull=[0.05]*len(top_users))
    fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(size=12)
    )

    return fig_bar, fig_pie