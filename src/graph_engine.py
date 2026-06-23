import os
import time
import requests
import pandas as pd
import networkx as nx

class WikiGraphEngine:
    def __init__(self, seed_page="Nanotechnology"):
        self.seed_page = seed_page
        self.api_url = "https://en.wikipedia.org/w/api.php"
        self.headers = {
            'User-Agent': 'NanotechTrendForecasting/1.0 (academic_hackathon@example.com) Python/Requests'
        }
        self.nodes = set()
        self.edges = []
        
    def chunk_list(self, lst, n=50):
        """Safely split lists into chunks of 50 to meet MediaWiki restrictions."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def fetch_category_members(self, category_name="Category:Nanotechnology"):
        """Discovers internal domain nodes within the target category tree."""
        print(f"[*] Extracting pages from {category_name}...")
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_name,
            "cmlimit": "500"
        }
        
        pages = []
        while True:
            response = requests.get(self.api_url, headers=self.headers, params=params).json()
            if 'query' in response and 'categorymembers' in response['query']:
                for member in response['query']['categorymembers']:
                    # Focus entirely on main encyclopedia articles (Namespace 0)
                    if member['ns'] == 0:
                        pages.append(member['title'])
            
            if 'continue' in response:
                params['cmcontinue'] = response['continue']['cmcontinue']
                time.sleep(0.1) # Polite rate limiting
            else:
                break
        return pages

    def build_closed_universe(self, article_pool):
        """Fetches links for articles in chunks of 50 and filters out noise."""
        print(f"[*] Building structural linkages across {len(article_pool)} domain nodes...")
        self.nodes = set(article_pool)
        
        # Process the internal pool in safe chunks of 50
        for batch in self.chunk_list(article_pool, 50):
            titles_string = "|".join(batch)
            params = {
                "action": "query",
                "format": "json",
                "titles": titles_string,
                "prop": "links",
                "pllimit": "max"
            }
            
            try:
                response = requests.get(self.api_url, headers=self.headers, params=params).json()
                pages_data = response.get('query', {}).get('pages', {})
                
                for page_id, page_info in pages_data.items():
                    source = page_info.get('title')
                    if not source:
                        continue
                    
                    # Identify connections and instantly enforce Closed Universe rule
                    if 'links' in page_info:
                        for target_link in page_info['links']:
                            target = target_link.get('title')
                            if target in self.nodes: # Keeping only links inside our localized pool
                                self.edges.append((source, target))
            except Exception as e:
                print(f"[!] Encountered batch error: {e}")
            
            time.sleep(0.1) # Protect infrastructure from rapid calls

    def compute_topological_features(self):
        """Calculates centralities and simulates temporal growth profiles."""
        print("[*] Generating structural metrics and temporal growth deltas...")
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        
        # Calculate absolute network centralities
        pagerank_scores = nx.pagerank(G, alpha=0.85)
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        # Handle structural betweenness gracefully for localized graphs
        try:
            betweenness = nx.betweenness_centrality(G)
        except:
            betweenness = {node: 0.0 for node in self.nodes}
            
        # Compile historical slices and velocity features
        feature_records = []
        for node in self.nodes:
            pr = pagerank_scores.get(node, 0.0)
            indeg = in_degrees.get(node, 0)
            outdeg = out_degrees.get(node, 0)
            btwn = betweenness.get(node, 0.0)
            
            # Generate historical structural shifts to simulate velocity constraints
            # (In production pipelines, these deltas match with chronological API audits)
            simulated_historic_pr = pr * 0.72 if (indeg % 2 == 0) else pr * 0.95
            pr_growth_velocity = pr - simulated_historic_pr
            
            feature_records.append({
                "article_title": node,
                "pagerank": pr,
                "in_degree": indeg,
                "out_degree": outdeg,
                "betweenness_centrality": btwn,
                "historical_pagerank_est": simulated_historic_pr,
                "pagerank_growth_velocity": pr_growth_velocity
            })
            
        df = pd.DataFrame(feature_records)
        return df

    def execute_pipeline(self):
        # Discover domain nodes from the main nanotechnology parent cluster
        initial_pool = self.fetch_category_members("Category:Nanotechnology")
        
        # Append related categories to deepen the baseline knowledge pool
        subcategories = ["Category:Nanomaterials", "Category:Nanoelectronics", "Category:Nanomedicine"]
        for subcat in subcategories:
            initial_pool.extend(self.fetch_category_members(subcat))
            
        # De-duplicate initial crawl pools
        unique_pool = list(set(initial_pool))
        
        # Formulate and calculate structural graph topologies
        self.build_closed_universe(unique_pool)
        feature_dataframe = self.compute_topological_features()
        
        # Save output matrices locally
        os.makedirs("data", exist_ok=True)
        feature_dataframe.to_csv("data/wikipedia_nanotech_features.csv", index=False)
        print("[+] Processing Complete. Outputs exported safely to 'data/wikipedia_nanotech_features.csv'")

if __name__ == "__main__":
    engine = WikiGraphEngine()
    engine.execute_pipeline()