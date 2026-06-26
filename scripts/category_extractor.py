"""
Extract Wikipedia categories and build category-based filtering dataset.
"""

import json
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path


def load_all_snapshots():
    """Load all yearly snapshots and extract categories."""
    snapshot_base = Path(__file__).parent.parent / "data-prep" / "Yearly snapshot data"
    
    # Map: title -> {year -> categories}
    title_categories = defaultdict(dict)
    
    for year in range(2020, 2027):
        snapshot_path = snapshot_base / f"{year}.json"
        
        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry in data:
                title = entry.get('title')
                categories = entry.get('categories', [])
                
                if title:
                    title_categories[title][year] = categories
        
        except Exception as e:
            print(f"Warning: Could not load {year} snapshot: {e}")
    
    return title_categories


def extract_all_categories(title_categories):
    """Extract all unique categories."""
    all_categories = set()
    
    for year_cats in title_categories.values():
        for categories in year_cats.values():
            all_categories.update(categories)
    
    # Clean category names
    cleaned = set()
    for cat in all_categories:
        # Remove Wikipedia prefixes and clean up
        clean = cat.replace('Wikipedia categories|', '').replace('_', ' ').strip()
        if clean and len(clean) > 2:  # Filter out very short names
            cleaned.add(clean)
    
    return sorted(cleaned)


def get_title_current_categories(title, title_categories, year=2026):
    """Get categories for a title in the current year."""
    if title not in title_categories:
        return []
    
    # Try to get current year, fallback to most recent
    if year in title_categories[title]:
        return title_categories[title][year]
    
    # Fallback to most recent year available
    available_years = sorted(title_categories[title].keys(), reverse=True)
    if available_years:
        return title_categories[title][available_years[0]]
    
    return []


def build_categories_json(predictions_df, title_categories):
    """
    Build categories.json for frontend filtering.
    
    Args:
        predictions_df: DataFrame with columns [title, page_id, prediction_score]
        title_categories: Dict mapping title -> {year -> [categories]}
    
    Returns:
        List of category records
    """
    # Create category -> node mapping
    category_to_nodes = defaultdict(list)
    
    for _, row in predictions_df.iterrows():
        title = row['title']
        page_id = row['page_id']
        score = row['prediction_score']
        
        categories = get_title_current_categories(title, title_categories)
        cleaned_categories = [
            cat.replace('Wikipedia categories|', '').replace('_', ' ').strip()
            for cat in categories
        ]
        
        for cat in cleaned_categories:
            if cat and len(cat) > 2:
                category_to_nodes[cat].append({
                    'title': title,
                    'page_id': page_id,
                    'prediction_score': float(score)
                })
    
    # Build category records
    categories_list = []
    
    for category_name in sorted(category_to_nodes.keys()):
        nodes = category_to_nodes[category_name]
        
        # Sort nodes by prediction score
        nodes.sort(key=lambda x: x['prediction_score'], reverse=True)
        
        category_record = {
            'category_name': category_name,
            'node_count': len(nodes),
            'average_prediction_score': float(sum(n['prediction_score'] for n in nodes) / len(nodes)),
            'top_predicted_nodes': nodes[:10]  # Top 10 in this category
        }
        
        categories_list.append(category_record)
    
    # Sort by average prediction score
    categories_list.sort(key=lambda x: x['average_prediction_score'], reverse=True)
    
    return categories_list


def get_category_index(categories_list):
    """Create a quick lookup index for categories."""
    return {cat['category_name']: idx for idx, cat in enumerate(categories_list)}
