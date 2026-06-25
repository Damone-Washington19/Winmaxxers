"""
Frontend Data Layer Builder

Generates all optimized JSON datasets for portal:
- node_lookup.json: Complete node database
- predictions.json: Ranked predictions
- communities.json: Community detection results
- categories.json: Category filtering data
- timeline.json: Historical trends
- feature_importance.json: Model explanation
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import builders
from community_detection import (
    load_2026_graph, 
    build_networkx_graph, 
    detect_communities,
    assign_community_ids,
    load_categories_from_snapshots,
    generate_community_label,
    create_communities_json
)
from category_extractor import (
    load_all_snapshots,
    build_categories_json
)
from timeline_builder import build_timeline_json


class FrontendDataBuilder:
    """Main builder class for frontend data layer."""
    
    def __init__(self, data_dir='data', output_dir='data'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.predictions_df = self._load_predictions()
        self.feature_importance_df = self._load_feature_importance()
        self.forecast_df = self._load_forecast()
        self.all_features = self._load_all_features()
        self.title_categories = load_all_snapshots()
        self.nodes_2026, self.edges_2026 = load_2026_graph()
    
    def _load_predictions(self):
        """Load 2027 predictions."""
        path = self.data_dir / 'predictions_2027.csv'
        df = pd.read_csv(path)
        df = df.sort_values('prediction_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        return df
    
    def _load_feature_importance(self):
        """Load feature importance."""
        path = self.data_dir / 'feature_importance.csv'
        return pd.read_csv(path)
    
    def _load_forecast(self):
        """Load 2027 forecast (1-year ahead)."""
        path = self.data_dir / 'forecast_2027.csv'
        return pd.read_csv(path)
    
    def _load_all_features(self):
        """Load all historical features."""
        path = Path(__file__).parent.parent / "data-prep" / "Features" / "all_years_features.csv"
        return pd.read_csv(path)
    
    def build_node_lookup(self):
        """Build node_lookup.json with all node details."""
        print("\n📍 Building node_lookup.json...")
        
        # Build community detection
        G = build_networkx_graph(self.nodes_2026, self.edges_2026)
        communities = detect_communities(G)
        community_assignment, num_communities = assign_community_ids(G, communities, self.nodes_2026)
        
        # Store for other methods
        self.community_assignment = community_assignment
        
        # Map node metrics from all_features for year 2026
        features_2026 = self.all_features[self.all_features['year'] == 2026]
        metrics_map = {}
        for _, row in features_2026.iterrows():
            metrics_map[row['title']] = row
        
        node_lookup = {}
        
        for idx, pred_row in self.predictions_df.iterrows():
            title = pred_row['title']
            page_id = pred_row['page_id']
            
            # Get node metrics from 2026 features
            node_metrics = metrics_map.get(title, {})
            comm_id = community_assignment.get(title, -1)
            
            # Get categories
            categories = self._get_current_categories(title)
            
            # Build feature contributions (top 3 features for this node)
            feature_contribs = self._get_feature_contributions(title)
            
            # Connected nodes (neighbors in graph)
            connected = self._get_connected_nodes(title, G, 10)
            
            node_record = {
                'title': title,
                'page_id': page_id,
                'prediction_score': float(pred_row['prediction_score']),
                'rank': int(pred_row['rank']),
                'community_id': int(comm_id),
                'community_label': generate_community_label([title], self.title_categories),
                'categories': categories,
                'pagerank': float(node_metrics.get('pagerank', pred_row.get('pagerank', 0))),
                'degree': int(node_metrics.get('total_degree', 0)),
                'authority_score': float(node_metrics.get('authority_score', pred_row.get('authority_score', 0))),
                'historical_metrics': {
                    'pagerank_trend': self._get_trend(page_id, 'pagerank'),
                    'degree_trend': self._get_trend(page_id, 'total_degree'),
                    'authority_trend': self._get_trend(page_id, 'authority_score'),
                },
                'connected_nodes': connected,
                'feature_contributions': feature_contribs
            }
            
            node_lookup[title] = node_record
        
        # Save
        output_path = self.output_dir / 'node_lookup.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(node_lookup, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created node_lookup.json with {len(node_lookup)} nodes")
        return node_lookup
    
    def build_predictions(self):
        """Build predictions.json for rankings."""
        print("\n📊 Building predictions.json...")
        
        predictions_list = []
        
        for idx, row in self.predictions_df.iterrows():
            categories = self._get_current_categories(row['title'])
            
            pred_record = {
                'title': row['title'],
                'page_id': row['page_id'],
                'prediction_score': float(row['prediction_score']),
                'rank': int(idx + 1),
                'community': int(self.community_assignment.get(row['title'], -1)) if hasattr(self, 'community_assignment') else -1,
                'categories': categories[:3],  # Top 3 categories
                'pagerank': float(row['pagerank']),
                'predicted_pagerank_next': float(row['predicted_pagerank_next']),
            }
            
            predictions_list.append(pred_record)
        
        # Save
        output_path = self.output_dir / 'predictions.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_list, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created predictions.json with {len(predictions_list)} ranked nodes")
        return predictions_list
    
    def build_communities(self):
        """Build communities.json."""
        print("\n🔗 Building communities.json...")
        
        # Build community detection
        G = build_networkx_graph(self.nodes_2026, self.edges_2026)
        communities = detect_communities(G)
        community_assignment, num_communities = assign_community_ids(G, communities, self.nodes_2026)
        
        # Store for other methods
        self.community_assignment = community_assignment
        
        # Map predictions to communities
        comm_predictions = defaultdict(list)
        for _, row in self.predictions_df.iterrows():
            comm_id = community_assignment.get(row['title'], -1)
            if comm_id >= 0:  # Only include predictions with valid community
                comm_predictions[comm_id].append(row)
        
        communities_list = []
        
        for comm_id, nodes in enumerate(communities):
            community_nodes = [title for title in nodes if title in community_assignment.values()]
            pred_scores = comm_predictions.get(comm_id, [])
            
            if not pred_scores:
                continue
            
            pred_df = pd.DataFrame(pred_scores)
            
            # Generate label from dominant categories
            label = generate_community_label(list(nodes), self.title_categories)
            
            # Top categories
            all_categories = []
            for node in nodes:
                all_categories.extend(self._get_current_categories(node))
            category_counter = defaultdict(int)
            for cat in all_categories:
                category_counter[cat] += 1
            top_categories = sorted(category_counter.items(), key=lambda x: x[1], reverse=True)[:5]
            top_categories = [cat[0] for cat in top_categories]
            
            # Top predicted nodes
            top_predicted = pred_df.nlargest(5, 'prediction_score')[['title', 'page_id', 'prediction_score']].to_dict('records')
            for item in top_predicted:
                item['prediction_score'] = float(item['prediction_score'])
            
            community_record = {
                'community_id': int(comm_id),
                'community_label': label,
                'node_count': len(nodes),
                'average_prediction_score': float(pred_df['prediction_score'].mean()),
                'top_categories': top_categories,
                'top_predicted_nodes': top_predicted
            }
            
            communities_list.append(community_record)
        
        # Sort by average prediction score
        communities_list.sort(key=lambda x: x['average_prediction_score'], reverse=True)
        
        # Save
        output_path = self.output_dir / 'communities.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(communities_list, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created communities.json with {len(communities_list)} communities")
        return communities_list
    
    def build_categories(self):
        """Build categories.json."""
        print("\n📑 Building categories.json...")
        
        categories_list = build_categories_json(self.predictions_df, self.title_categories)
        
        # Save
        output_path = self.output_dir / 'categories.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(categories_list, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created categories.json with {len(categories_list)} categories")
        return categories_list
    
    def build_timeline(self):
        """Build timeline.json."""
        print("\n📈 Building timeline.json...")
        
        timeline_data = build_timeline_json(self.predictions_df)
        
        # Save
        output_path = self.output_dir / 'timeline.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created timeline.json with {len(timeline_data)} node timelines")
        return timeline_data
    
    def build_feature_importance(self):
        """Build feature_importance.json."""
        print("\n🔬 Building feature_importance.json...")
        
        fi_data = {
            'features': [],
            'model_type': 'XGBRegressor',
            'target_variable': 'pagerank_growth_clipped',
            'description': 'Feature importance for predicting topic growth'
        }
        
        for idx, row in self.feature_importance_df.iterrows():
            fi_data['features'].append({
                'rank': int(idx + 1),
                'feature_name': row['feature'],
                'importance': float(row['importance']),
                'relative_importance': float(row['importance'] / self.feature_importance_df['importance'].sum())
            })
        
        # Save
        output_path = self.output_dir / 'feature_importance.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fi_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created feature_importance.json with {len(fi_data['features'])} features")
        return fi_data
    
    # Helper methods
    
    def _get_current_categories(self, title):
        """Get categories for a title."""
        if title not in self.title_categories:
            return []
        
        # Get most recent year available
        year_cats = self.title_categories[title]
        available_years = sorted(year_cats.keys(), reverse=True)
        
        if available_years:
            cats = year_cats[available_years[0]]
            return [
                cat.replace('Wikipedia categories|', '').replace('_', ' ').strip()
                for cat in cats if cat
            ][:10]  # Top 10
        
        return []
    
    def _get_trend(self, page_id, metric):
        """Get historical trend for a metric."""
        node_hist = self.all_features[self.all_features['page_id'] == page_id].sort_values('year')
        
        if len(node_hist) < 2:
            return None
        
        first_val = node_hist.iloc[0][metric] if metric in node_hist.columns else None
        last_val = node_hist.iloc[-1][metric] if metric in node_hist.columns else None
        
        if first_val is None or pd.isna(first_val) or first_val == 0:
            return None
        
        growth = (last_val - first_val) / first_val if pd.notna(last_val) else None
        
        return {
            'first_year': int(node_hist.iloc[0]['year']),
            'last_year': int(node_hist.iloc[-1]['year']),
            'first_value': float(first_val),
            'last_value': float(last_val),
            'growth': float(growth) if growth is not None else None
        }
    
    def _get_feature_contributions(self, title):
        """Get top features contributing to prediction for a node."""
        # This would require per-node SHAP values or similar
        # For now, return top global features
        return self.feature_importance_df.head(3)[['feature', 'importance']].to_dict('records')
    
    def _get_connected_nodes(self, title, G, limit=10):
        """Get connected nodes (neighbors) in graph."""
        connected = []
        
        if title in G:
            neighbors = list(G.neighbors(title))[:limit]
            
            for neighbor in neighbors:
                pred = self.predictions_df[self.predictions_df['title'] == neighbor]
                if len(pred) > 0:
                    connected.append({
                        'title': neighbor,
                        'page_id': int(pred.iloc[0]['page_id']),
                        'prediction_score': float(pred.iloc[0]['prediction_score'])
                    })
        
        return connected
    
    def build_all(self):
        """Build all frontend data files."""
        print("\n" + "="*60)
        print("🚀 FRONTEND DATA LAYER BUILDER")
        print("="*60)
        print("\n📊 Model forecasts: 2026 → 2027 (1-year ahead)")
        print("   Not multi-year recursive forecasting\n")
        
        try:
            # Build in order (some depend on others)
            self.build_communities()
            self.build_predictions()
            self.build_node_lookup()
            self.build_categories()
            self.build_timeline()
            self.build_feature_importance()
            
            print("\n" + "="*60)
            print("✅ ALL FRONTEND DATA FILES GENERATED SUCCESSFULLY")
            print("="*60)
            print(f"\n📁 Output directory: {self.output_dir.absolute()}")
            print("📅 Forecast: 2026 → 2027 (1-year ahead prediction)")
            print("\nGenerated files:")
            for file in sorted(self.output_dir.glob('*.json')):
                size_kb = file.stat().st_size / 1024
                print(f"  ✓ {file.name} ({size_kb:.1f} KB)")
            
        except Exception as e:
            print(f"\n❌ Error building frontend data: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    import sys
    
    # Allow custom output directory
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    
    builder = FrontendDataBuilder(data_dir='data', output_dir=output_dir)
    builder.build_all()


if __name__ == '__main__':
    main()
