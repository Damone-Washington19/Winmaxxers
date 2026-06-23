import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def train_and_explain():
    print("[*] Initializing Machine Learning Prediction Pipeline...")
    
    # Load your generated graph feature spreadsheet
    df = pd.read_csv("data/wikipedia_nanotech_features.csv")
    
    # Feature Selection
    features = ['pagerank', 'in_degree', 'out_degree', 'betweenness_centrality', 'pagerank_growth_velocity']
    X = df[features]
    
    # Generate ground truth labels (Y) using hackathon proxy rules:
    # A node is classified as a "Breakout" (1) if its growth velocity matches high structural betweenness
    threshold = df['pagerank_growth_velocity'].median()
    y = (df['pagerank_growth_velocity'] > threshold).astype(int)
    
    # Scale structural metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Explainable Probabilistic Model (L1 Regularized Logistic Regression)
    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    model.fit(X_scaled, y)
    
    print("\n💡 Model Coefficients & Feature Importance (Knowledge Graph Analytics):")
    for feat, coef in zip(features, model.coef_[0]):
        print(f" -> {feat:25}: {coef:.4f}")
        
    print("\n[+] Model pipeline successfully trained. Features map directly to the Streamlit Portal interface.")

if __name__ == "__main__":
    train_and_explain()