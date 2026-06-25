# 🔬 NanoTrend Vanguard: Wikipedia Knowledge Graph Engine

**Track Alignment:** This project is officially submitted to **Track 2: What’s Next? — Future Event Predictor**.

**NanoTrend Vanguard** structures a localized **"Closed Universe" Network Graph** extracted directly from Wikipedia’s core nanotechnology clusters (Nanotechnology, Nanomaterials, Nanoelectronics, and Nanomedicine). By isolating outbound node linkages exclusively to targets residing within this baseline discovery pool, the pipeline processes structural matrices free from arbitrary global network distortion. Through a **multi-year historical analysis (2022–2026)**, the system applies machine learning to topological shifts, translating graph evolution into high-fidelity **future trend probabilities**.

---

## 🛠️ Repository Architecture & File Directory

```text
Winmaxxers/
├── .github/workflows/
│   └── ci.yml                # CI/CD pipeline for automated dependency and code compilation validation.
├── Model/
│   ├── Final_Model (1).ipynb # R&D notebook for XGBoost training and time-series cross-validation.
│   ├── predict_future.py     # Script to generate high-fidelity next-year PageRank growth forecasts.
│   ├── recursive_forecast.py # Multi-year (2027–2031) recursive trend projection engine.
│   ├── train_model.py        # Core logic for training XGBoost regression and classification models.
│   └── utils.py              # Helper functions for data ingestion and feature engineering.
├── app/
│   └── app.py                # Streamlit-based web portal and dark-mode interactive UI layer.
├── data-prep/
│   ├── Features/
│   │   ├── merge.py          # Utility to consolidate yearly topological metrics into a master dataset.
│   │   └── temporal_analysis.py # Logic for computing growth velocity, momentum, and acceleration.
│   ├── Graphs/
│   │   ├── featuresExtract.py # Orchestrator for extracting 15+ network centrality metrics per year.
│   │   └── make_graph.py     # Visualization tool for generating static snapshots of the domain graph.
│   ├── Yearly snapshot data/
│   │   ├── constructGraphs.py # Reconstructs yearly networks from historical JSON snapshots.
│   │   └── master_pages.json # Defines the "Closed Universe" discovery pool of articles.
│   ├── WikiScrape.py         # Wikipedia API scraper for node and category tree discovery.
│   └── generate_snapshot_data.py # Historical version retrieval logic for multi-year analysis.
├── src/
│   ├── graph_builder.py      # Script for network construction and connectivity validation.
│   └── graph_engine.py       # Main engine for category extraction and link reconstruction.
├── Wikipedia Predictor Roadmap.pdf # Strategic project overview and delivery milestones.
├── requirements.txt          # Defined project dependencies for environment setup.
└── README.md                 # Comprehensive project documentation (this file).
```

---

## 🧮 Mathematical Engine & Feature Definitions

The engine synthesizes complex network dynamics into predictive signals across several key dimensions:

*   **Current Importance:** **PageRank (PR)** is calculated via iterative power methods ($\alpha = 0.85$) to isolate domain prestige, while **In-Degree Centrality** tracks the raw volume of incoming citations within the specialized cluster.
*   **Strategic Obscurity:** **Betweenness Centrality** serves as our primary proxy for identifying future breakouts; it tracks sparse connectivity tunnels that bridge disparate paradigms, highlighting nodes that act as critical information brokers despite low current prestige.
*   **PageRank Growth Velocity:** This metric tracks the acceleration of a topic's relevance by evaluating **historical topology velocity changes**, capturing momentum that raw PageRank fails to identify.

### Data Integrity & Validation
To ensure analytical fidelity, the system includes a rigorous validation pipeline:
*   **Connectivity Checks:** The engine computes **Network Density** and performs **Weak Connectivity checks** to ensure a coherent directed graph structure.
*   **GCC Extraction:** The system isolates the **Giant Connected Component (GCC)** to strip out isolated noise and orphan nodes that could skew probabilities.
*   **Community Detection:** Utilizing **Girvan-Newman and Louvain clustering**, the engine uncovers ambiguous hidden relationships, mapping structural modularity as an explicit feature for the machine learning model.

---

## 🖥️ Web Portal & Interactive Visualization

The **NanoTrend Vanguard Portal** provides a professional-grade gateway for exploring predictive scientific data:

*   **Architecture:** Built on a **Streamlit** framework for high-performance data serving and a **PyVis** engine for rendering complex network topologies.
*   **Interactive Graphics:** The portal renders dark-mode interactive graphs where **node sizing** scales dynamically with **PageRank** (absolute importance).
*   **Breakout Identification:** Node coloring serves as a strategic alert system; **high-betweenness nodes** with accelerating growth momentum are highlighted in **red** to flag them as high-probability structural breakout candidates for the 2026–2030 period.

---

## 🚀 Local Quickstart Guide & Installation

Follow these steps to deploy the engine and portal locally:

1.  **Clone and Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Installs: `networkx`, `pandas`, `xgboost`, `streamlit`, and `pyvis`)*.

2.  **Generate Feature Matrix:**
    Execute the core engine to reconstruct the graph and extract topological features:
    ```bash
    python src/graph_engine.py
    ```
    *(Outputs `data/wikipedia_nanotech_features.csv`)*.

3.  **Validate Graph and Detect Communities:**
    Run the validation pipeline to optimize for the GCC and apply clustering layers:
    ```bash
    python src/graph_builder.py
    ```
    *(Updates matrix with `structural_cluster_id` signals)*.

4.  **Launch the Interactive Portal:**
    ```bash
    streamlit run app/app.py
    ```
    *(Launches the web interface for trend exploration)*.

---

## 🏆 Award Alignment Matrix

| Award Category | Technical Implementation | Judging Alignment |
| :--- | :--- | :--- |
| **Knowledge Graph Analytics** | Integration of 15+ topological features, including **Betweenness Centrality** for obscurity and **Growth Velocity** for momentum. | Demonstrates elite-level analysis of scientific knowledge evolution through advanced graph theory. |
| **Portal Award** | Streamlit-based **dark-mode UI** featuring real-time **PyVis interactive graphs** and dynamic breakout probability rankings. | Delivers a sophisticated, user-centric gateway for exploring complex predictive nanotechnology data. |
| **Gateways Impact** | Robust, automated **Wikipedia snapshot engine** and a validated **CI/CD pipeline** for reproducible results. | Provides a highly reproducible framework for community-led future scientific trend prediction. |
