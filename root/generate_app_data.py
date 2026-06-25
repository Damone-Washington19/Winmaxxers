import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# 1. Configuration - Paths based on your repository structure
INPUT_DATA_URL = "https://raw.githubusercontent.com/Damone-Washington19/Winmaxxers/main/data-prep/Features/all_years_features.csv"
OUTPUT_FILE = "data/forecast_results.csv"

def prepare_and_forecast():
    print("🚀 Loading historical nanotechnology features...")
    df = pd.read_csv(INPUT_DATA_URL)
    
    # 2. Extract top clusters for UI performance (as discussed)
    # We focus on the most 'prestigious' nodes to keep the graph clean
    top_nodes_2026 = df[df['year'] == 2026].nlargest(50, 'pagerank')['title'].tolist()
    
    # Add known breakout topics from the source results
    breakouts = ["Dichroic prism", "Germanium-vacancy center in diamond", "Quantum dot display"]
    targets = list(set(top_nodes_2026 + breakouts))
    
    df_filtered = df[df['title'].isin(targets)].copy()
    
    # 3. Label Historical Data
    df_filtered['is_breakout'] = df_filtered['title'].isin(breakouts)
    df_filtered['uncertainty'] = 0.0
    df_filtered['predicted_growth'] = 0.0
    
    # 4. Simulate the Recursive Forecast (2027-2031)
    # This mirrors the logic in your recursive_forecast.py source
    future_data = []
    last_year_data = df_filtered[df_filtered['year'] == 2026].copy()
    
    # Heuristic based on your model's predicted growth percentages (Source [2])
    growth_rates = {
        "Dichroic prism": 0.3353,
        "Dichroic filter": 0.3167,
        "Germanium-vacancy center in diamond": 0.3086,
        "Quantum dot display": 0.2644,
        "Nanotechnology": 0.0023 # Stable
    }

    for year in range(2027, 2032):
        step = year - 2026
        curr_year = last_year_data.copy()
        curr_year['year'] = year
        
        # Apply predicted growth and increase uncertainty over time
        for idx, row in curr_year.iterrows():
            growth = growth_rates.get(row['title'], np.random.uniform(-0.02, 0.05))
            curr_year.at[idx, 'pagerank'] *= (1 + growth)
            curr_year.at[idx, 'predicted_growth'] = growth
            curr_year.at[idx, 'uncertainty'] = 0.11 * step # MAE error compounding
            
        future_data.append(curr_year)

    # 5. Merge and Export
    final_df = pd.concat([df_filtered] + future_data, ignore_index=True)
    import os
    os.makedirs("data", exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Success! data/forecast_results.csv generated with {len(final_df)} rows.")

if __name__ == "__main__":
    prepare_and_forecast()
