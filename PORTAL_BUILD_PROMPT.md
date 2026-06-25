# Interactive Portal Build Prompt
## NanoVerse: An Award-Winning Knowledge Intelligence Platform

---

## VISION

Build an **interactive, exploratory knowledge portal** that presents emerging nanotechnology trends as a JARVIS-style intelligence system. Users explore a constellation of topics, discover connections, understand predictions, and learn how the AI arrived at conclusions.

**Aesthetic:** Scientific + futuristic + accessible. Think NASA mission control meets modern tech UI.

**Tone:** Authoritative yet welcoming. Educational without being pedantic.

---

## PART 1: INDEX.HTML RESTRUCTURE

### Current State
The current index.html uses Streamlit (Stlite) for Python execution. This is **too slow and cumbersome** for real-time interaction.

### New Approach
Keep Streamlit but **optimize for speed and elegance**:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NanoVerse — Emerging Topic Intelligence Platform</title>
    
    <!-- Design System -->
    <style>
        :root {
            --primary: #00E5A0;      /* Accent green */
            --primary-dark: #00B87D;
            --secondary: #00E5FF;    /* Cyan */
            --accent: #FFD700;       /* Gold for rankings */
            --success: #00E5A0;
            --warning: #FFA500;
            --danger: #FF5252;
            --bg-dark: #0A0E27;      /* Deep blue-black */
            --bg-card: #1A1F3A;      /* Slightly lighter */
            --border: #2D3561;
            --text: #E8EAED;
            --text-dim: #9CA3AF;
            --text-bright: #FFFFFF;
        }
        
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            margin: 0;
            overflow-x: hidden;
        }
    </style>
    
    <!-- D3.js for graph visualization (optional, can use Plotly via Python) -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    
    <!-- Streamlit app -->
    <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.58.0/build/stlite.js"></script>
</head>
<body>
    <div id="stlite"></div>
    
    <script>
        stlite.mount(
            {
                requirements: [
                    "pandas",
                    "plotly",
                    "networkx",
                    "scikit-learn"
                ],
                entrypoint: "app.py",
                files: {
                    "app.py": { url: "app.py" },
                    // Data files served locally
                    "data/predictions.json": { url: "data/predictions.json" },
                    "data/node_lookup.json": { url: "data/node_lookup.json" },
                    "data/communities.json": { url: "data/communities.json" },
                    "data/categories.json": { url: "data/categories.json" },
                    "data/timeline.json": { url: "data/timeline.json" },
                    "data/feature_importance.json": { url: "data/feature_importance.json" }
                }
            },
            document.getElementById("stlite")
        );
    </script>
</body>
</html>
```

---

## PART 2: APP.PY COMPREHENSIVE ARCHITECTURE

### Overview Structure

```python
"""
NanoVerse Portal — Interactive Emerging Topic Intelligence

Pages:
1. HOME: Dashboard with key stats and top predictions
2. EXPLORER: Interactive knowledge universe (graph + search)
3. RANKINGS: 2027 Emerging topics ranked by prediction score
4. DEEP DIVES: Community and category exploration
5. HOW IT WORKS: Model explanation with feature importance
6. ABOUT: Data source and methodology
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict

# ===== CONFIGURATION =====
st.set_page_config(
    page_title="NanoVerse — Emerging Topic Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== DESIGN SYSTEM =====
COLORS = {
    "primary": "#00E5A0",
    "secondary": "#00E5FF", 
    "accent": "#FFD700",
    "success": "#00E5A0",
    "warning": "#FFA500",
    "danger": "#FF5252",
    "bg_dark": "#0A0E27",
    "bg_card": "#1A1F3A",
    "border": "#2D3561",
    "text": "#E8EAED",
    "text_dim": "#9CA3AF",
}

# ===== CUSTOM CSS =====
st.markdown(f"""
<style>
    /* Hide Streamlit elements */
    [data-testid="stDecoration"] {{ display: none; }}
    
    /* Page background */
    .main {{ 
        background: linear-gradient(135deg, {COLORS['bg_dark']} 0%, #1A1B2E 100%);
        padding: 2rem;
    }}
    
    /* Metric cards */
    .metric-card {{
        background: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
    }}
    
    /* Titles */
    h1, h2, h3 {{ color: {COLORS['text_bright']}; }}
    
    /* Links */
    a {{ color: {COLORS['primary']}; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
</style>
""", unsafe_allow_html=True)

# ===== DATA LOADING (with caching) =====

@st.cache_resource
def load_data():
    """Load all frontend JSON files efficiently."""
    data = {}
    
    files = {
        'predictions': 'data/predictions.json',
        'node_lookup': 'data/node_lookup.json',
        'communities': 'data/communities.json',
        'categories': 'data/categories.json',
        'timeline': 'data/timeline.json',
        'feature_importance': 'data/feature_importance.json'
    }
    
    for key, path in files.items():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)
        except FileNotFoundError:
            st.error(f"❌ Missing: {path}")
            data[key] = None
    
    return data

# Load all data at startup
DATA = load_data()

# ===== UTILITY FUNCTIONS =====

def get_trend_color(growth_value):
    """Return color based on predicted growth."""
    if growth_value > 0.02:
        return COLORS['success']
    elif growth_value > -0.02:
        return COLORS['secondary']
    else:
        return COLORS['danger']

def get_trend_badge(growth_value):
    """Return trend badge emoji + label."""
    if growth_value > 0.02:
        return "🔺 Rising", COLORS['success']
    elif growth_value > -0.02:
        return "→ Stable", COLORS['secondary']
    else:
        return "🔻 Declining", COLORS['danger']

def format_metric(value, format_type='default'):
    """Format numbers for display."""
    if format_type == 'percent':
        return f"{value*100:.2f}%"
    elif format_type == 'rank':
        return f"#{int(value)}"
    elif format_type == 'decimal':
        return f"{value:.4f}"
    return str(value)

# ===== PAGE: HOME =====

def page_home():
    """Dashboard with key statistics and top predictions."""
    
    st.markdown("# 🌌 NanoVerse")
    st.markdown("### Emerging Topic Intelligence Platform")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Intelligence from Knowledge.**
        
        Predict emerging trends in nanotechnology by analyzing Wikipedia's knowledge graph.
        Discover topics gaining momentum, understand why they're predicted to rise, explore connections.
        
        📅 **Forecast:** 2026 → 2027 (1-year prediction)  
        📊 **Nodes:** 1,530 topics tracked  
        🌐 **Communities:** 342 clusters detected  
        📂 **Categories:** 6,129 Wikipedia categories mapped
        """)
    
    with col2:
        # Key stats boxes
        predictions = DATA['predictions']
        top_growth = predictions[0]['prediction_score']
        top_topic = predictions[0]['title']
        
        st.metric(
            "🏆 Top Prediction",
            top_topic[:30],
            f"+{top_growth*100:.2f}%"
        )
    
    # Top emerging topics carousel
    st.markdown("---")
    st.markdown("## 🚀 Top Emerging Topics (2027)")
    
    top_n = 10
    cols = st.columns(5)
    
    for idx, pred in enumerate(DATA['predictions'][:top_n]):
        if idx % 5 == 0:
            cols = st.columns(5)
        
        with cols[idx % 5]:
            trend_badge, color = get_trend_badge(pred['prediction_score'])
            
            st.markdown(f"""
            <div style='
                background: {COLORS['bg_card']};
                border: 2px solid {color};
                border-radius: 8px;
                padding: 1rem;
                text-align: center;
            '>
                <div style='font-weight: bold; margin-bottom: 0.5rem;'>
                    #{pred['rank']}
                </div>
                <div style='font-size: 13px; margin-bottom: 0.5rem;'>
                    {pred['title'][:25]}...
                </div>
                <div style='color: {color}; font-weight: bold; font-size: 14px;'>
                    {trend_badge}
                </div>
                <div style='color: {COLORS['text_dim']}; font-size: 11px; margin-top: 0.5rem;'>
                    Growth: +{pred['prediction_score']*100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Community highlights
    st.markdown("---")
    st.markdown("## 🔗 Top Communities")
    
    cols = st.columns(3)
    for idx, comm in enumerate(DATA['communities'][:3]):
        with cols[idx]:
            st.markdown(f"""
            **{comm['community_label']}**
            
            • {comm['node_count']} nodes
            • Avg prediction: +{comm['average_prediction_score']*100:.2f}%
            • Top category: {comm['top_categories'][0] if comm['top_categories'] else 'N/A'}
            """)
    
    # Category highlights
    st.markdown("---")
    st.markdown("## 📑 Top Categories")
    
    cols = st.columns(3)
    for idx, cat in enumerate(DATA['categories'][:3]):
        with cols[idx]:
            st.markdown(f"""
            **{cat['category_name']}**
            
            • {cat['node_count']} nodes
            • Avg prediction: +{cat['average_prediction_score']*100:.2f}%
            """)

# ===== PAGE: EXPLORER =====

def page_explorer():
    """Interactive knowledge universe with search and filtering."""
    
    st.markdown("# 🔍 Knowledge Explorer")
    st.markdown("Search, filter, and discover emerging topics")
    
    # Search interface
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_term = st.text_input(
            "Search topics...",
            placeholder="e.g., 'graphene', 'quantum', 'nanotube'"
        )
    
    with search_col2:
        filter_type = st.selectbox(
            "Filter by:",
            ["All", "Rising", "Stable", "Declining"]
        )
    
    # Search results
    if search_term:
        results = [
            node for node in DATA['node_lookup'].values()
            if search_term.lower() in node['title'].lower()
        ]
        
        if results:
            st.markdown(f"**Found {len(results)} matching topics**")
            
            for node in results[:20]:  # Show top 20
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{node['title']}**")
                    st.caption(f"Community: {node['community_label']} | Categories: {', '.join(node['categories'][:2])}")
                
                with col2:
                    trend_badge, _ = get_trend_badge(node['prediction_score'])
                    st.markdown(trend_badge)
                
                with col3:
                    st.metric("Score", f"+{node['prediction_score']*100:.2f}%")
                
                # Detail expander
                with st.expander("Details"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Rank", node['rank'])
                    with col_b:
                        st.metric("PageRank", f"{node['pagerank']:.6f}")
                    with col_c:
                        st.metric("Degree", node['degree'])
                    
                    st.markdown("**Categories:**")
                    st.write(", ".join(node['categories']))
                    
                    # Timeline chart
                    if node['title'] in DATA['timeline']:
                        timeline = DATA['timeline'][node['title']]
                        years = list(timeline['years'].keys())
                        pageranks = [
                            timeline['years'][year].get('pagerank', 0) 
                            for year in years
                        ]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=years,
                            y=pageranks,
                            mode='lines+markers',
                            name='PageRank',
                            line=dict(color=COLORS['primary'], width=3),
                            marker=dict(size=8)
                        ))
                        fig.update_layout(
                            title="Historical PageRank",
                            template="plotly_dark",
                            height=300,
                            showlegend=False,
                            paper_bgcolor=COLORS['bg_card'],
                            plot_bgcolor=COLORS['bg_dark']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
        else:
            st.info("No matching topics found")
    else:
        st.info("📝 Enter a search term to explore topics")
    
    # Browse by community
    st.markdown("---")
    st.markdown("## Browse by Community")
    
    selected_comm = st.selectbox(
        "Select a community:",
        options=[c['community_label'] for c in DATA['communities'][:20]],
        key="comm_select"
    )
    
    if selected_comm:
        comm_data = next(c for c in DATA['communities'] if c['community_label'] == selected_comm)
        
        st.markdown(f"**{selected_comm}**")
        st.markdown(f"• {comm_data['node_count']} nodes | Avg prediction: +{comm_data['average_prediction_score']*100:.2f}%")
        
        st.markdown("**Top predicted topics in this community:**")
        for node in comm_data['top_predicted_nodes'][:10]:
            st.write(f"• **{node['title']}** (+{node['prediction_score']*100:.2f}%)")

# ===== PAGE: RANKINGS =====

def page_rankings():
    """2027 predictions ranked by emergence score."""
    
    st.markdown("# 🏆 2027 Emerging Topic Rankings")
    st.markdown("Topics predicted to gain momentum in the next year")
    
    # Sorting options
    col1, col2, col3 = st.columns(3)
    with col1:
        sort_by = st.selectbox("Sort by:", ["Prediction Score ↓", "Alphabetical"])
    with col2:
        limit = st.slider("Show top N:", 10, 100, 30)
    with col3:
        view = st.radio("View:", ["List", "Table"], horizontal=True)
    
    # Sort data
    preds = DATA['predictions'].copy()
    if sort_by == "Alphabetical":
        preds.sort(key=lambda x: x['title'])
    
    preds = preds[:limit]
    
    if view == "List":
        for pred in preds:
            col1, col2, col3, col4 = st.columns([0.5, 2, 1, 0.8])
            
            with col1:
                # Gold medal for top 3
                if pred['rank'] <= 3:
                    st.markdown(f"<div style='font-size: 24px; text-align: center;'>{'🥇' if pred['rank']==1 else '🥈' if pred['rank']==2 else '🥉'}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='font-size: 14px; text-align: center; color: {COLORS['text_dim']};'>#{pred['rank']}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{pred['title']}**")
                st.caption(f"Category: {', '.join(pred['categories'][:2])}")
            
            with col3:
                trend_badge, _ = get_trend_badge(pred['prediction_score'])
                st.markdown(trend_badge)
            
            with col4:
                st.metric("Score", f"+{pred['prediction_score']*100:.2f}%")
            
            st.divider()
    
    else:  # Table view
        df = pd.DataFrame([
            {
                'Rank': f"#{p['rank']}",
                'Topic': p['title'],
                'Score': f"+{p['prediction_score']*100:.2f}%",
                'PageRank': f"{p['pagerank']:.6f}",
                'Categories': ', '.join(p['categories'][:2])
            }
            for p in preds
        ])
        st.dataframe(df, use_container_width=True)

# ===== PAGE: DEEP DIVES =====

def page_deep_dives():
    """Community and category deep exploration."""
    
    st.markdown("# 🌐 Deep Dives")
    
    dive_type = st.radio("Explore by:", ["Communities", "Categories"], horizontal=True)
    
    if dive_type == "Communities":
        st.markdown("## Community Analysis")
        
        selected = st.selectbox(
            "Select community:",
            options=[c['community_label'] for c in DATA['communities'][:30]],
            key="deep_comm"
        )
        
        comm = next(c for c in DATA['communities'] if c['community_label'] == selected)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes", comm['node_count'])
        with col2:
            st.metric("Avg Prediction", f"+{comm['average_prediction_score']*100:.2f}%")
        with col3:
            st.metric("Top Category", comm['top_categories'][0] if comm['top_categories'] else "N/A")
        
        st.markdown("**Top Categories:**")
        st.write(" • ".join(comm['top_categories']))
        
        st.markdown("**Top Predicted Nodes:**")
        for node in comm['top_predicted_nodes']:
            st.write(f"• {node['title']} (+{node['prediction_score']*100:.2f}%)")
    
    else:  # Categories
        st.markdown("## Category Analysis")
        
        selected = st.selectbox(
            "Select category:",
            options=[c['category_name'] for c in DATA['categories'][:50]],
            key="deep_cat"
        )
        
        cat = next(c for c in DATA['categories'] if c['category_name'] == selected)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", cat['node_count'])
        with col2:
            st.metric("Avg Prediction", f"+{cat['average_prediction_score']*100:.2f}%")
        
        st.markdown("**Top Predicted Topics:**")
        for node in cat['top_predicted_nodes']:
            st.write(f"• {node['title']} (+{node['prediction_score']*100:.2f}%)")

# ===== PAGE: HOW IT WORKS =====

def page_how_it_works():
    """Model explanation and transparency."""
    
    st.markdown("# 🔬 How It Works")
    st.markdown("Understanding the prediction model")
    
    # Model overview
    st.markdown("## The Prediction Pipeline")
    
    st.markdown("""
    ### 1️⃣ Data Collection (2020-2026)
    - Wikipedia graph snapshots for 7 years
    - Extract graph features: PageRank, centrality, degree, authority
    - Create temporal feature database
    
    ### 2️⃣ Target Variable: Growth
    - Calculate year-over-year PageRank growth
    - Growth = (PageRank_t+1 - PageRank_t) / PageRank_t
    - Predict topics with highest growth momentum
    
    ### 3️⃣ Model Training
    - Time-series cross-validation (no leakage)
    - XGBoost regression for continuous predictions
    - Train on 2020-2025, test on 2026
    - Generate 2027 forecast
    
    ### 4️⃣ Prediction Output
    - **Prediction Score:** Predicted growth rate for 2027
    - **Uncertainty:** ±{0.098:.1%} (1-year forecast horizon)
    - **Ranking:** Top topics most likely to emerge
    """)
    
    # Feature importance
    st.markdown("---")
    st.markdown("## 📊 Feature Importance")
    st.markdown("What drives the predictions?")
    
    if DATA['feature_importance']:
        features = DATA['feature_importance']['features']
        
        # Top features visualization
        top_features = features[:10]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f['feature_name'] for f in reversed(top_features)],
            x=[f['importance'] for f in reversed(top_features)],
            orientation='h',
            marker=dict(color=COLORS['primary']),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top 10 Predictive Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template="plotly_dark",
            height=400,
            paper_bgcolor=COLORS['bg_card'],
            plot_bgcolor=COLORS['bg_dark'],
            margin=dict(l=200)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature explanations
        st.markdown("### What Each Feature Means")
        feature_explanations = {
            "pagerank": "Importance/centrality score in the network",
            "in_degree": "Number of incoming links (citations)",
            "out_degree": "Number of outgoing links (references)",
            "betweenness_centrality": "How often topic bridges between communities",
            "closeness_centrality": "Average distance to other topics",
            "eigenvector_centrality": "Connected to other important topics",
            "authority_score": "Hub/authority in the knowledge graph",
            "clustering_coefficient": "How tightly connected its neighbors are",
        }
        
        for feature in top_features[:5]:
            name = feature['feature_name']
            explanation = feature_explanations.get(name, "Network metric")
            st.markdown(f"**{name}** — {explanation}")

# ===== PAGE: ABOUT =====

def page_about():
    """Project information and data sources."""
    
    st.markdown("# ℹ️ About NanoVerse")
    
    st.markdown("""
    ## What is NanoVerse?
    
    An **emerging topic prediction platform** that identifies which nanotechnology topics 
    are gaining momentum on Wikipedia. Built by analyzing 7 years of knowledge graph evolution,
    trained with machine learning, presented with AI transparency.
    
    ## Data Sources
    
    - **Wikipedia Knowledge Graph** — 2020-2026 network snapshots
    - **Graph Metrics** — PageRank, centrality scores, community detection
    - **Machine Learning** — XGBoost regression with time-series CV
    - **Forecast** — 2026 → 2027 (1-year ahead predictions)
    
    ## Key Statistics
    
    | Metric | Value |
    |--------|-------|
    | Nodes Tracked | 1,530 topics |
    | Time Period | 2020-2026 (7 years) |
    | Communities | 342 detected clusters |
    | Categories | 6,129 Wikipedia categories |
    | Features | 18 graph metrics |
    | Prediction Horizon | 2027 (1-year) |
    | Model Type | XGBoost Regression |
    | Uncertainty | ±9.8% |
    
    ## How Predictions Work
    
    1. **Extract Features** — Calculate graph metrics for each topic
    2. **Target Variable** — Measure year-over-year growth momentum
    3. **Train Model** — Learn patterns from 2020-2025 data
    4. **Forecast** — Predict 2027 growth for each topic
    5. **Rank Results** — Sort by emergence score
    
    ## Award-Winning Elements
    
    ✨ **Elegant UI** — Dark theme with neon accents (scientific aesthetic)
    🎯 **Clear Purpose** — Know exactly what you're looking at and why
    📊 **Full Transparency** — See feature importance, understand predictions
    ⚡ **Performance** — Precomputed data, zero-latency interactions
    🔍 **Explorable** — Search, filter, drill down into details
    
    ---
    
    **Built with:** Streamlit + Python + XGBoost + Wikipedia  
    **Data Generated:** 2026-06-25  
    **Forecast Validity:** 2027 (1-year ahead)
    """)

# ===== MAIN APP ROUTER =====

def main():
    """Main app with page navigation."""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🌌 NanoVerse")
        
        page = st.radio(
            "Navigate:",
            [
                "🏠 Home",
                "🔍 Explorer",
                "🏆 Rankings",
                "🌐 Deep Dives",
                "🔬 How It Works",
                "ℹ️ About"
            ]
        )
    
    # Page routing
    if "Home" in page:
        page_home()
    elif "Explorer" in page:
        page_explorer()
    elif "Rankings" in page:
        page_rankings()
    elif "Deep Dives" in page:
        page_deep_dives()
    elif "How It Works" in page:
        page_how_it_works()
    elif "About" in page:
        page_about()

if __name__ == "__main__":
    main()
```

---

## PART 3: OPTIMIZATION & PERFORMANCE GUIDELINES

### Data Loading Strategy
```python
# Cache ALL data at startup (runs once)
@st.cache_resource
def load_data():
    # Load all JSON files
    # Return as dict for instant access
    
# Use for every interaction — no latency
node = DATA['node_lookup'][title]
```

### Search Optimization
```python
# Precompute search index
@st.cache_resource
def build_search_index():
    index = {}
    for node in DATA['node_lookup'].values():
        for word in node['title'].lower().split():
            if word not in index:
                index[word] = []
            index[word].append(node)
    return index

# Fast lookup during search
results = search_index.get(search_term.lower(), [])
```

### Filtering & Sorting
- All filtering happens in Python (in-memory)
- Sort predictions by score, alpha, rank on-the-fly
- Category/community filters use precomputed mappings

---

## PART 4: AWARD-WINNING FEATURES

### Visual Hierarchy
- **Hero Section** — Large, compelling statement
- **Key Metrics** — Critical numbers prominently displayed
- **Rankings** — Gold medals for top 3
- **Depth** — Expandable details without clutter

### Color Psychology
- **Green (#00E5A0)** — Growth, emergence, optimism
- **Gold (#FFD700)** — Achievement, top rank
- **Cyan (#00E5FF)** — Technology, intelligence
- **Red (#FF5252)** — Decline, warning

### Interactive Elements
- Expandable details (no page reloads)
- Real-time search (no delay)
- Smooth transitions
- Informative tooltips

### Educational Content
- Explain model pipeline step-by-step
- Show feature importance visually
- Provide context for every metric
- Link predictions to data

---

## PART 5: DEPLOYMENT CHECKLIST

- [ ] Data files copied to `/data/` directory
- [ ] `index.html` updated with new styling
- [ ] `app.py` complete with all pages
- [ ] Sidebar navigation working
- [ ] Search functionality fast
- [ ] Charts rendering with Plotly
- [ ] Mobile responsive (Streamlit handles this)
- [ ] All pages accessible
- [ ] No console errors

---

## TECHNICAL NOTES

### Why Streamlit + Plotly?
- **Streamlit** — Rapid Python app development
- **Plotly** — Beautiful, interactive charts
- **JSON Data** — Instant access, no DB needed
- **Caching** — Zero-latency interactions

### File Size Optimization
The JSON files are large (13.7 MB total) but:
- Loaded once at startup (cache)
- Streamed from `/data/` directory
- Streamlit handles compression
- User sees instant interactions

### Future Enhancements
1. **Network Graph Visualization** — D3.js or Plotly network for relationships
2. **Real-time Updates** — If model retrains, auto-refresh data
3. **Export Features** — Download rankings as CSV/PDF
4. **Comparison Mode** — Compare 2025 vs 2026 vs 2027 predictions
5. **Alerts** — Notify when a topic reaches threshold

---

## SUCCESS METRICS

Your portal is award-winning when:

✅ Users understand the model in 60 seconds  
✅ Search returns results instantly  
✅ Every metric has clear context  
✅ Predictions feel legitimate and explained  
✅ Design feels premium and scientific  
✅ Navigation is intuitive  
✅ Data is always accessible  
✅ Errors are handled gracefully  

---

## NEXT STEPS

1. **Replace `index.html`** with new version (minimal changes needed)
2. **Rewrite `app.py`** following the architecture above
3. **Verify data loading** — ensure all JSON files accessible
4. **Test all pages** — navigate through every feature
5. **Optimize images/fonts** — ensure fast load times
6. **Deploy** — Streamlit Cloud, Heroku, or self-hosted

---

**Goal:** A portal that feels like mission control for emerging trends. Where every click teaches you something new.
