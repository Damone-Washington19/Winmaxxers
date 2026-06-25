import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import time

# --- Page Configuration ---
st.set_page_config(page_title="NanoTrend Vanguard Portal", layout="wide")

# --- Sidebar: Confidence & Explanations ---
with st.sidebar:
    st.title("🛡️ Model Confidence")
    # Accuracy values from Source [1]
    st.metric("Global Accuracy (R²)", "0.97", help="How well the AI matches historical trends.")
    st.metric("Avg. Error (MAE)", "0.11", help="This measures the 'noise' in predictions.")
    
    st.divider()
    
    st.subheader("Explanation Style")
    style = st.radio("Choose a level:", ["Middle School", "Professional", "Technical"])
    
    if style == "Middle School":
        st.info("**The Popularity Contest:** We track who's making new friends on Wikipedia to guess who will be a superstar next year!")
    elif style == "Professional":
        st.info("**Structural Momentum:** This model measures topological shifts in science paradigms to forecast domain prestige.")
    else:
        st.info("**Recursive XGBoost:** Regressing on lagged graph centralities (PageRank, Betweenness) to predict temporal growth deltas.")

# --- Title ---
st.title("🔬 Nanotechnology Knowledge Graph Engine")
st.markdown("Explore the evolution of nanotechnology from **2020** through predicted breakthroughs in **2031**.")

# --- Step 4 Integration: Loading Real Data ---
@st.cache_data
def load_real_data():
    try:
        df = pd.read_csv("data/forecast_results.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure Step 3 was completed successfully.")
        return None

df_full = load_real_data()

if df_full is not None:
    # --- Timeline & Playback Control ---
    years = sorted(df_full['year'].unique())
    
    if 'year_index' not in st.session_state:
        st.session_state.year_index = 0

    col_slider, col_play = st.columns([2, 3])
    selected_year = col_slider.select_slider("Select Year", options=years, value=years[st.session_state.year_index])

    if col_play.button("▶ Play Evolution"):
        for i in range(len(years)):
            st.session_state.year_index = i
            st.rerun()
            time.sleep(0.4)

    # --- Filtering Logic ---
    # We focus on top nodes for UI performance [4]
    df_year = df_full[df_full['year'] == selected_year].copy()
    
    # --- UI Feature: Hover Information ---
    with st.expander("🔍 Top Projected Growth Topics for " + str(selected_year)):
        growth_display = df_year.sort_values('pagerank', ascending=False).head(5)
        st.table(growth_display[['title', 'pagerank', 'uncertainty']])

    # --- Graph Visualization (Pyvis) ---
    net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")

    for _, row in df_year.iterrows():
        # Size nodes by PageRank (Prestige) [5, 6]
        size = row['pagerank'] * 10000 
        
        # Color logic: Red for breakouts, Blue for established topics
        # Sources identified "Dichroic prism" and "Quantum dot display" as breakouts [7]
        color = "#ff4b4b" if row['is_breakout'] else "#00f0ff"
        
        # Add node with metadata for tooltips
        label = row['title']
        status = "🚀 BREAKOUT" if row['is_breakout'] else "Stable"
        tooltip = f"Prestige (PageRank): {row['pagerank']:.4f}\nStatus: {status}\nModel Uncertainty: {row['uncertainty']:.2f}"
        
        net.add_node(label, label=label, size=size, color=color, title=tooltip)

    # Simplified connections for visualization
    nodes = df_year['title'].tolist()
    for i in range(len(nodes)-1):
        if i % 3 == 0: # Create a "web" appearance rather than a single line
            net.add_edge(nodes[i], nodes[min(i+3, len(nodes)-1)])

    # Render
    net.save_graph("temp_graph.html")
    with open("temp_graph.html", 'r', encoding='utf-8') as f:
        components.html(f.read(), height=620)
    
    st.success(f"Displaying {selected_year} Scientific Landscape. Model R²: 0.97.")

else:
    st.warning("Waiting for forecast_results.csv to be generated in the /data folder...")
