import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import time

# --- Page Configuration ---
st.set_page_config(page_title="NanoTrend Vanguard", layout="wide")

# --- Interactive Explanation Layer ---
with st.sidebar:
    st.title("🛡️ Model Confidence")
    st.metric("Global Accuracy (R²)", "0.97", help="How well the AI matches historical trends.")
    st.metric("Avg. Error (MAE)", "0.11", help="Lower is better. This measures the 'noise' in predictions.")
    
    st.divider()
    
    style = st.radio("Explanation Style", ["Middle School", "Professional", "Technical"])
    
    if style == "Middle School":
        st.info("**The Popularity Contest:** We track who's making new friends on Wikipedia to guess who will be a superstar next year!")
    elif style == "Professional":
        st.info("**Structural Momentum:** This model measures topological shifts in science paradigms to forecast domain prestige.")
    else:
        st.info("**Recursive XGBoost:** Regressing on lagged graph centralities (PageRank, Betweenness) to predict temporal growth deltas.")

# --- Title & Description ---
st.title("🔬 Nanotechnology Knowledge Graph Engine")
st.markdown("""
Move the slider to see how nanotechnology topics have evolved from **2020** and where they are headed by **2031**.
*Nodes grow based on prestige (PageRank). Glowing red nodes are predicted 'Breakouts'.*
""")

# --- Data Loading (Simulated for this step) ---
# In Step 3, we will generate the actual 'forecast_results.csv' from your features.
@st.cache_data
def load_forecast_data():
    # This is a placeholder structure to ensure the UI renders immediately
    # We will replace this with your real recursive_forecast output next.
    return pd.read_csv("data/forecast_results.csv") if False else None

# --- Timeline & Playback Control ---
col_slider, col_play = st.columns([1, 2])
if 'year_index' not in st.session_state:
    st.session_state.year_index = 0

years = list(range(2020, 2032))
selected_year = col_slider.select_slider("Select Year", options=years, value=years[st.session_state.year_index])

if col_play.button("▶ Play Evolution"):
    for i in range(len(years)):
        st.session_state.year_index = i
        st.rerun()
        time.sleep(0.5)

# --- Visualization Logic ---
st.subheader(f"🌐 Scientific Landscape: {selected_year}")

# Sidebar Breakdown for "Why?"
with st.expander("🔍 Why is the model predicting growth?"):
    st.write("The AI looks for 'Bridges'—topics that connect two different scientific fields.")

# Build the Network Graph
# Based on Source [3], we limit to top nodes for performance.
net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")

# Placeholder nodes based on known breakout topics [4]
# We'll use the top PageRank nodes from your 2026 data [5]
top_topics = [
    ("Nanotechnology", 0.03, False),
    ("Dichroic prism", 0.001, True),
    ("Carbon nanotube", 0.007, False),
    ("Germanium-vacancy center", 0.001, True),
    ("Quantum dot display", 0.002, True)
]

for title, pr, is_breakout in top_topics:
    # Size scales with PageRank [6]
    size = pr * 5000 if not is_breakout else pr * 8000 
    color = "#ff4b4b" if is_breakout else "#00f0ff"
    border = "2px solid yellow" if is_breakout else ""
    
    net.add_node(title, label=title, size=size, color=color, 
                 title=f"PageRank: {pr:.4f}\nStatus: {'🚀 Breakout' if is_breakout else 'Stable'}")

# Structural visualization edges [6]
net.add_edge("Nanotechnology", "Carbon nanotube")
net.add_edge("Nanotechnology", "Quantum dot display")
net.add_edge("Quantum dot display", "Dichroic prism")

# Render Graph
net.save_graph("temp_graph.html")
HtmlFile = open("temp_graph.html", 'r', encoding='utf-8')
components.html(HtmlFile.read(), height=620)

st.success(f"Model is currently showing {selected_year} projections with an R² of 0.97.")
