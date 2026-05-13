import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(page_title="Graph Analytics Dashboard", layout="wide")

st.markdown("""
<style>
            
/* Make tabs full width */
.stTabs [data-baseweb="tab-list"] {
    width: 100%;
    justify-content: space-between;
}

/* Each tab takes equal space */
.stTabs [data-baseweb="tab"] {
    flex-grow: 1;
    text-align: center; 
    font-weight: bold;
    padding: 10px 0px;
}
            
/* TAB TEXT (important target) */
.stTabs [data-baseweb="tab"] p {
    font-size: 20px !important;  
    margin: 0;
}

/* Active tab highlight */
.stTabs [aria-selected="true"] {
    background-color:  #2f2f2f;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Link Analytics using Wiki-Vote Dataset ")
st.write("Analysis using PageRank, HITS, and Degree Centrality")

# ----------------------------------------
# LOAD DATA
# ----------------------------------------
@st.cache_data
def load_data():
    nodes = pd.read_csv("nodes.csv")
    edges = pd.read_csv("edges.csv")
    return nodes, edges

nodes_df, edges_df = load_data()

# ----------------------------------------
# DATASET OVERVIEW
# ----------------------------------------
st.subheader("📁 Dataset Overview")
col1, col2 = st.columns(2)
col1.metric("Total Nodes", len(nodes_df))
col2.metric("Total Edges", len(edges_df))

with st.expander("View Sample Data"):
    st.dataframe(nodes_df.head(10))
    st.dataframe(edges_df.head(10))

# ----------------------------------------
# BUILD GRAPH
# ----------------------------------------
G = nx.DiGraph()

for _, row in nodes_df.iterrows():
    G.add_node(row["id"], name=row["name"])

for _, row in edges_df.iterrows():
    G.add_edge(row["source"], row["target"])

# ----------------------------------------
# PAGERANK
# ----------------------------------------
pagerank_scores = nx.pagerank(G, alpha=0.85)

pagerank_df = pd.DataFrame(
    [{"id": node, "pagerank_score": score} for node, score in pagerank_scores.items()]
)

pagerank_df = pagerank_df.merge(nodes_df, on="id")
pagerank_df = pagerank_df.sort_values(by="pagerank_score", ascending=False).reset_index(drop=True)
pagerank_df["rank"] = pagerank_df.index + 1

# ----------------------------------------
# HITS
# ----------------------------------------
hubs, authorities = nx.hits(G, max_iter=200)

hits_df = pd.DataFrame({
    "id": list(hubs.keys()),
    "hub_score": list(hubs.values()),
    "authority_score": list(authorities.values())
})

hits_df = hits_df.merge(nodes_df, on="id")
hits_df = hits_df.sort_values(by="authority_score", ascending=False).reset_index(drop=True)
hits_df["rank"] = hits_df.index + 1

# ----------------------------------------
# DEGREE
# ----------------------------------------
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())

deg_df = pd.DataFrame({
    "id": list(in_deg.keys()),
    "in_degree": list(in_deg.values()),
    "out_degree": list(out_deg.values())
})

deg_df = deg_df.merge(nodes_df, on="id")
deg_df = deg_df.sort_values(by="in_degree", ascending=False).reset_index(drop=True)
deg_df["rank"] = deg_df.index + 1

# ----------------------------------------
# TABS NAVIGATION
# ----------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 PageRank",
    "🔗 HITS",
    "📌 Degree",
    "🌐 Graph",
    "📈 Analysis"
])

# ----------------------------------------
# TAB 1: PAGERANK
# ----------------------------------------
with tab1:
    st.subheader("🏆 Top 20 Users by PageRank")
    st.dataframe(pagerank_df.head(20), use_container_width=True)

    st.subheader("📊 Top 10 PageRank")
    top10 = pagerank_df.head(10).copy()
    top10["id"] = top10["id"].astype(str)
    st.bar_chart(top10.set_index("id")["pagerank_score"])

    st.subheader("🔍 Search a User's PageRank")

    user_id_pr = st.number_input("Enter User ID", min_value=int(nodes_df["id"].min()), max_value=int(nodes_df["id"].max()), step=1, key="pagerank_input")

    if st.button("Find User Rank", key="btn_pagerank"):
        result = pagerank_df[pagerank_df["id"] == user_id_pr]

        if not result.empty:
            st.success("User Found!")
            st.dataframe(result, use_container_width=True)
        else:
            st.error("User ID not found")

    # ----------------------------------------
    # DOWNLOAD
    # ----------------------------------------
    st.subheader("⬇ Download Results")

    csv = pagerank_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download PageRank CSV",
        data=csv,
        file_name="pagerank_results.csv",
        mime="text/csv"
    )

# ----------------------------------------
# TAB 2: HITS
# ----------------------------------------
with tab2:
    st.subheader("🏆 Top 20 Authority Scores")
    st.dataframe(hits_df.head(20), use_container_width=True)

    st.subheader("📊 Top 10 Authority")
    top10 = hits_df.head(10).copy()
    top10["id"] = top10["id"].astype(str)
    st.bar_chart(top10.set_index("id")["authority_score"])

    # ----------------------------------------
    # DOWNLOAD HITS
    # ----------------------------------------
    csv_hits = hits_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download HITS CSV",
        csv_hits,
        "hits.csv",
        "text/csv"
    )

# ----------------------------------------
# TAB 3: DEGREE
# ----------------------------------------
with tab3:
    st.subheader("🏆 Top 10 Users by In-Degree")
    st.dataframe(deg_df.head(10), use_container_width=True)

    st.subheader("🔍 Find Degree of User")
    user_id_deg = st.number_input("Enter User ID", min_value=int(nodes_df["id"].min()), max_value=int(nodes_df["id"].max()), step=1, key="deg_input")

    if st.button("Calculate Degree", key="btn_deg"):
        if user_id_deg in in_deg:
            user_name = nodes_df[nodes_df["id"] == user_id_deg]["name"].values[0]

            st.success(f"User: {user_name}")

            col1, col2 = st.columns(2)
            col1.metric("In-Degree", in_deg[user_id_deg])
            col2.metric("Out-Degree", out_deg[user_id_deg])
        else:
            st.error("User not found")

    # ----------------------------------------
    # DOWNLOAD DEGREE
    # ----------------------------------------
    csv_deg = deg_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Degree CSV",
        csv_deg,
        "degree.csv",
        "text/csv"
    )

# ----------------------------------------
# TAB 4: GRAPH
# ----------------------------------------
with tab4:
    st.subheader("🌐 Graph Visualization")

    basis = st.selectbox(
        "Select Ranking Basis",
        ["PageRank", "HITS Authority", "In-Degree"]
    )

    top_n = st.slider("Top N Nodes", 5, 50, 10)

    if basis == "PageRank":
        top_nodes = pagerank_df.head(top_n)["id"].tolist()
    elif basis == "HITS Authority":
        top_nodes = hits_df.head(top_n)["id"].tolist()
    else:
        top_nodes = deg_df.head(top_n)["id"].tolist()

    subG = G.subgraph(top_nodes)

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(subG, seed=42)

    nx.draw(subG, pos, with_labels=True, node_size=600, font_size=8, ax=ax)

    st.pyplot(fig)

# ----------------------------------------
# TAB 5: ANALYSIS
# ----------------------------------------
with tab5:
    st.subheader("📊 Algorithm Comparison")

    comparison_df = pagerank_df[["id", "pagerank_score"]]
    comparison_df = comparison_df.merge(
        hits_df[["id", "authority_score", "hub_score"]], on="id"
    )
    comparison_df = comparison_df.merge(
        deg_df[["id", "in_degree"]], on="id"
    )

    st.dataframe(comparison_df.head(20))

    # Heatmap
    st.subheader("📈 Correlation Heatmap")

    corr = comparison_df[
        ["pagerank_score", "authority_score", "hub_score", "in_degree"]
    ].corr()

    fig, ax = plt.subplots(figsize=(3, 2))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 5},
        ax=ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=6)

    st.pyplot(fig)

    # Top-K overlap
    st.subheader("🔄 Top-K Overlap")

    k = st.slider("Select K", 5, 50, 10)

    top_pr = set(pagerank_df.head(k)["id"])
    top_hits = set(hits_df.head(k)["id"])

    common = top_pr.intersection(top_hits)

    st.write("Common Nodes:", list(common))
    st.write("Count:", len(common))

    # User analysis
    st.subheader("🔍 User Analysis")

    user_id = st.number_input("Enter User ID", min_value=int(nodes_df["id"].min()), max_value=int(nodes_df["id"].max()), step=1, key="analysis_input")


    if st.button("Analyze User", key="btn_analysis"):
        if user_id in comparison_df["id"].values:
            row = comparison_df[comparison_df["id"] == user_id].iloc[0]

            user_name = nodes_df[nodes_df["id"] == user_id]["name"].values[0]
            st.success(f"User: {user_name}")

            col1, col2, col3 = st.columns(3)
            col1.metric("PageRank", round(row["pagerank_score"], 6))
            col2.metric("Authority", round(row["authority_score"], 6))
            col3.metric("Hub", round(row["hub_score"], 6))

            st.metric("In-Degree", int(row["in_degree"]))
        else:
            st.error("User not found")

