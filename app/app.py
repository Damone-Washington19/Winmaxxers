import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="NanoTrend Portal", layout="wide")

st.title("🔬 NanoTrend Vanguard Portal")
st.subheader("Predicting the Future of Nanotechnology via Knowledge Graph Topology")

# Load engineered data matrix
@st.cache_data
def load_data():
    return pd.read_csv("data/wikipedia_nanotech_features.csv")

try:
    df = load_data()
    
    # Simple simulated predictive model for the hackathon UI
    # Combines high bridging (betweenness) with accelerating momentum
    df['Breakout_Probability'] = (df['betweenness_centrality'] * 0.4 + df['pagerank_growth_velocity'] * 0.6)
    df['Breakout_Probability'] = (df['Breakout_Probability'] - df['Breakout_Probability'].min()) / (df['Breakout_Probability'].max() - df['Breakout_Probability'].min())
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🚀 Top 2026–2030 Predicted Breakouts")
        st.write("Concepts currently obscure in PageRank but showing extreme structural topology shifts.")
        
        predictions = df.sort_values(by='Breakout_Probability', ascending=False).head(10)
        st.dataframe(predictions[['article_title', 'Breakout_Probability']].style.format({'Breakout_Probability': '{:.2%}'}))
        
    with col2:
        st.markdown("### 🕸️ Interactive Domain Topology")
        
        # Build quick visualization for top interconnected nodes
        net = Network(height="500px", width="100%", bgcolor="#1a1a1a", font_color="white")
        top_nodes = df.sort_values(by='pagerank', ascending=False).head(30)
        
        for _, row in top_nodes.iterrows():
            # Size node by its PageRank, color by its breakout potential
            color = "#ff4b4b" if row['Breakout_Probability'] > 0.7 else "#00f0ff"
            net.add_node(row['article_title'], label=row['article_title'], size=row['pagerank']*5000, color=color)
            
        # Add a few structural visualization edges
        titles = top_nodes['article_title'].tolist()
        for i in range(len(titles)-1):
            net.add_edge(titles[i], titles[i+1])
            
        net.save_graph("temp_network.html")
        HtmlFile = open("temp_network.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=520)

except Exception as e:
    st.error(f"Could not load feature matrix. Please run 'python src/graph_engine.py' first. Error: {e}")