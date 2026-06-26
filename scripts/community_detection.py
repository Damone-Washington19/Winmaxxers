"""
Community detection utilities using the 2026 graph.
Extract communities and generate labels from dominant categories.
"""

import pandas as pd
import numpy as np
import json
from collections import Counter
import networkx as nx
from pathlib import Path


def load_2026_graph():
    """Load 2026 nodes and edges."""
    base_path = Path(__file__).parent.parent / "data-prep" / "Graphs"
    
    nodes_df = pd.read_csv(base_path / "2026_nodes.csv")
    edges_df = pd.read_csv(base_path / "2026_edges.csv")
    
    return nodes_df, edges_df


def build_networkx_graph(nodes_df, edges_df):
    """Build NetworkX graph from nodes and edges."""
    G = nx.Graph()
    
    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        G.add_node(row['title'])
    
    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'])
    
    return G


def detect_communities(G, method='louvain'):
    """
    Detect communities using Louvain method (if networkx_community available).
    Falls back to greedy modularity optimization.
    """
    try:
        import networkx.algorithms.community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(G))
    except:
        # Fallback: simple degree-based clustering
        communities = _simple_community_detection(G)
    
    return communities


def _simple_community_detection(G):
    """
    Fallback: simple community detection using connected components.
    """
    components = list(nx.connected_components(G))
    return [set(comp) for comp in components]


def assign_community_ids(G, communities, nodes_df):
    """Assign community IDs to each node."""
    node_to_community = {}
    
    for comm_id, community in enumerate(communities):
        for node in community:
            node_to_community[node] = comm_id
    
    # Create mapping with missing nodes
    community_assignment = {}
    for _, row in nodes_df.iterrows():
        title = row['title']
        community_assignment[title] = node_to_community.get(title, -1)
    
    return community_assignment, len(communities)


def load_categories_from_snapshots():
    """Load categories from 2026 snapshot."""
    snapshot_path = Path(__file__).parent.parent / "data-prep" / "Yearly snapshot data" / "2026.json"
    
    with open(snapshot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Map title -> categories
    title_to_categories = {}
    for entry in data:
        title = entry.get('title')
        categories = entry.get('categories', [])
        if title:
            title_to_categories[title] = categories
    
    return title_to_categories


def generate_community_label(community_members, title_to_categories):
    """
    Generate community label from dominant categories.
    """
    category_counter = Counter()
    
    for member in community_members:
        categories = title_to_categories.get(member, {})
        # Handle both {year: [cats]} and just [cats] formats
        if isinstance(categories, dict):
            # Get most recent year
            for year_cats in categories.values():
                if isinstance(year_cats, list):
                    for cat in year_cats:
                        if isinstance(cat, str) and cat:
                            category_counter[cat] += 1
        elif isinstance(categories, list):
            for cat in categories:
                if isinstance(cat, str) and cat:
                    category_counter[cat] += 1
    
    if category_counter:
        top_category, _ = category_counter.most_common(1)[0]
        # Clean up category name
        if isinstance(top_category, str):
            label = top_category.replace('Wikipedia categories|', '').replace('_', ' ')
            return label[:50]  # Truncate if too long
    
    return "Unknown"


def create_communities_json(predictions_df):
    """
    Main function to create communities.json.
    Requires: predictions_df with columns [title, page_id, prediction_score, community_id]
    """
    nodes_df, edges_df = load_2026_graph()
    G = build_networkx_graph(nodes_df, edges_df)
    
    # Detect communities
    communities = detect_communities(G)
    community_assignment, num_communities = assign_community_ids(G, communities, nodes_df)
    
    # Load categories
    title_to_categories = load_categories_from_snapshots()
    
    # Build communities.json
    communities_list = []
    
    for comm_id in range(num_communities):
        community_nodes = [title for title, cid in community_assignment.items() if cid == comm_id]
        
        # Get stats for this community
        community_preds = predictions_df[predictions_df['title'].isin(community_nodes)]
        
        if len(community_preds) == 0:
            continue
        
        # Generate label from categories
        label = generate_community_label(community_nodes, title_to_categories)
        
        # Extract top categories for this community
        all_categories = []
        for node in community_nodes:
            all_categories.extend(title_to_categories.get(node, []))
        category_counter = Counter(all_categories)
        top_categories = [cat for cat, _ in category_counter.most_common(5)]
        
        # Get top predicted nodes in this community
        top_predicted = community_preds.nlargest(5, 'prediction_score')[['title', 'page_id', 'prediction_score']].to_dict('records')
        
        community_record = {
            'community_id': int(comm_id),
            'community_label': label,
            'node_count': len(community_nodes),
            'average_prediction_score': float(community_preds['prediction_score'].mean()),
            'top_categories': top_categories,
            'top_predicted_nodes': top_predicted
        }
        
        communities_list.append(community_record)
    
    # Sort by average prediction score descending
    communities_list.sort(key=lambda x: x['average_prediction_score'], reverse=True)
    
    return communities_list
