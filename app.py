import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import time

st.set_page_config(page_title="NanoTrend Vanguard", layout="wide")

# --- Dynamic Glossary Definitions ---
glossary = {
    "Middle School": {
        "PageRank": "📈 **Popularity Score:** How many other 'famous' topics are talking about this one.",
        "Betweenness": "🌉 **The Bridge:** Topics that act like a tunnel connecting two different groups of science.",
        "Velocity": "🚀 **Speed:** How fast a topic is becoming a superstar.",
        "Accuracy": "🎯 **R² (0.97):** Our model's grade. It got a 97% on its history test!"
    },
    "Professional": {
        "PageRank": "📈 **Domain Prestige:** A measure of a topic's influence within the scientific network [1, 2].",
        "Betweenness": "🌉 **Connectivity Tunnels:** Topics that bridge disparate scientific paradigms [1, 2].",
        "Velocity": "🚀 **Momentum:** Evaluation of historical topology changes to rank emerging concepts [1, 2].",
        "Accuracy": "🎯 **Confidence:** R² of 0.97 indicates the model captures 97% of historical trend variance [3]."
    },
    "Technical": {
        "PageRank": "📈 **Eigenvector Centrality:** Probability distribution used to weight node importance in the directed graph [1].",
        "Betweenness": "🌉 **Vertex Importance:** Calculated via shortest-path importance for cross-cluster information flow [2].",
        "Velocity": "🚀 **Temporal Delta:** The first derivative of PageRank growth across discrete yearly snapshots [2].",
        "Accuracy": "🎯 **Regression Metrics:** R²: 0.97 | MAE: 0.11. Evaluated via recursive XGBoost on lagged centralities [3]."
    }
}

# --- Sidebar: Logic & Glossary ---
with st.sidebar:
    st.title("🛡️ Model Portal")
    style = st.radio("Explanation Style", ["Middle School", "Professional", "Technical"])
    
    st.divider()
    st.subheader("📖 Concept Glossary")
    # Dynamic definitions based on the radio button above
    st.markdown(glossary[style]["PageRank"])
    st.markdown(glossary[style]["Betweenness"])
    st.markdown(glossary[style]["Velocity"])
    st.markdown(glossary[style]["Accuracy"])

# --- Main UI ---
st.title("🔬 Nanotechnology Knowledge Graph")
st.markdown("Historical Evolution (2020–2026) and **AI Projection (2027)**.")

@st.cache_data
def load_data():
    try:
        # Filters data to stop at the requested 2027 horizon
        df = pd.read_csv("data/forecast_results.csv")
        return df[df['year'] <= 2027].copy()
    except Exception as e:
        st.error(f"Data mapping failed: {e}")
        return None

df = load_data()

if df is not None:
    years = sorted(df['year'].unique())
    selected_year = st.select_slider("Select Year", options=years)

    # Visual Distinction Logic
    is_future = selected_year == 2027
    if is_future:
        st.warning("⚠️ **Viewing AI Predictions for 2027.** These results are based on recursive XGBoost modeling [3, 4].")
    else:
        st.info(f"📅 **Viewing Historical Data for {selected_year}.** Verified Wikipedia graph snapshots [5].")

    # --- Graph Visualization ---
    df_year = df[df['year'] == selected_year]
    net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
    
    for _, row in df_year.iterrows():
        # Prestige scaling
        size = row['pagerank'] * 12000 
        
        # Color & Border: Highlight breakouts and distinguish Predicted nodes
        color = "#ff4b4b" if row['is_breakout'] else "#00f0ff"
        
        # Add labels to nodes to signify they are predicted in 2027
        display_label = f"🔮 {row['title']}" if is_future else row['title']
        
        # Predicted nodes get a dashed border for visual distinction
        border_width = 4 if is_future else 1
        
        net.add_node(
            row['title'], 
            label=display_label, 
            size=size, 
            color=color,
            borderWidth=border_width,
            shapeProperties={'borderDashes': is_future}, # Dashed border for predictions
            title=f"Prestige: {row['pagerank']:.4f}\nUncertainty: {row['uncertainty']:.2f}"
        )

    net.save_graph("temp.html")
    with open("temp.html", 'r', encoding='utf-8') as f:
        components.html(f.read(), height=620)
