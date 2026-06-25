"""
Build historical timeline data for node trend visualization.
"""

import pandas as pd
from collections import defaultdict
from pathlib import Path


def load_all_years_features():
    """Load features for all years."""
    features_path = Path(__file__).parent.parent / "data-prep" / "Features" / "all_years_features.csv"
    df = pd.read_csv(features_path)
    return df


def build_timeline_json(predictions_df):
    """
    Build timeline.json with historical metrics for each node.
    
    Args:
        predictions_df: Current year predictions
    
    Returns:
        Dict mapping title -> historical metrics by year
    """
    # Load all historical features
    all_features = load_all_years_features()
    
    timeline_data = {}
    
    # For each predicted node, collect historical metrics
    for _, pred_row in predictions_df.iterrows():
        title = pred_row['title']
        page_id = pred_row['page_id']
        
        # Get all records for this page across years
        node_history = all_features[all_features['page_id'] == page_id].sort_values('year')
        
        if len(node_history) == 0:
            continue
        
        # Build year -> metrics mapping
        years_data = {}
        
        for _, row in node_history.iterrows():
            year = int(row['year'])
            
            years_data[str(year)] = {
                'pagerank': float(row['pagerank']) if pd.notna(row['pagerank']) else None,
                'in_degree': int(row['in_degree']) if pd.notna(row['in_degree']) else None,
                'out_degree': int(row['out_degree']) if pd.notna(row['out_degree']) else None,
                'total_degree': int(row['total_degree']) if pd.notna(row['total_degree']) else None,
                'authority_score': float(row['authority_score']) if pd.notna(row['authority_score']) else None,
                'hub_score': float(row['hub_score']) if pd.notna(row['hub_score']) else None,
                'betweenness_centrality': float(row['betweenness_centrality']) if pd.notna(row['betweenness_centrality']) else None,
                'closeness_centrality': float(row['closeness_centrality']) if pd.notna(row['closeness_centrality']) else None,
                'eigenvector_centrality': float(row['eigenvector_centrality']) if pd.notna(row['eigenvector_centrality']) else None,
                'clustering_coefficient': float(row['clustering_coefficient']) if pd.notna(row['clustering_coefficient']) else None,
                'katz_centrality': float(row['katz_centrality']) if pd.notna(row['katz_centrality']) else None,
            }
        
        timeline_data[title] = {
            'page_id': page_id,
            'title': title,
            'years': years_data
        }
    
    return timeline_data


def calculate_growth_rate(timeline_entry):
    """Calculate historical growth rate for a node."""
    years_data = timeline_entry['years']
    
    if len(years_data) < 2:
        return None
    
    years_sorted = sorted(years_data.keys())
    first_year = years_data[years_sorted[0]]
    last_year = years_data[years_sorted[-1]]
    
    if first_year.get('pagerank') is None or last_year.get('pagerank') is None:
        return None
    
    pr_growth = (last_year['pagerank'] - first_year['pagerank']) / first_year['pagerank']
    
    return {
        'first_year': int(years_sorted[0]),
        'last_year': int(years_sorted[-1]),
        'pagerank_growth': float(pr_growth),
        'degree_growth': (last_year['total_degree'] - first_year['total_degree']) / max(1, first_year['total_degree'])
    }
