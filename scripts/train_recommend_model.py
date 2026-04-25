"""
Train a real Collaborative Filtering recommendation model on the 
Kaggle Amazon Cell Phones Reviews dataset.
Saves the product catalog, user mappings, and SVD embeddings to disk.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

print("=" * 60)
print("NEXUS-AI Recommender Training — Real Amazon Dataset")
print("=" * 60)

DATA_DIR = "/app/data/recommend"
OUT_DIR = "/app/models/recommend"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
print("[1/5] Loading real Amazon reviews & items...")
df_items = pd.read_csv(f"{DATA_DIR}/20191226-items.csv")
df_reviews = pd.read_csv(f"{DATA_DIR}/20191226-reviews.csv")

# Clean reviews (drop missing user names or ratings)
df_reviews = df_reviews.dropna(subset=['name', 'rating', 'asin'])

# Deduplicate item metadata
df_items = df_items.drop_duplicates(subset=['asin']).set_index('asin')

print(f"  Total Reviews : {len(df_reviews):,}")
print(f"  Total Items   : {len(df_items):,}")
print(f"  Unique Users  : {df_reviews['name'].nunique():,}")

# ── 2. Create User/Item Mappings ─────────────────────────────────────────────
print("[2/5] Creating user and item mappings...")
# Filter to items that actually have reviews and metadata
valid_asins = set(df_reviews['asin']).intersection(set(df_items.index))
df_reviews = df_reviews[df_reviews['asin'].isin(valid_asins)]

# Convert user names to IDs
user_names = df_reviews['name'].unique()
user2id = {name: i for i, name in enumerate(user_names)}
id2user = {i: name for name, i in user2id.items()}

# Convert ASINs to item IDs
item_asins = list(valid_asins)
item2id = {asin: i for i, asin in enumerate(item_asins)}
id2item = {i: asin for asin, i in item2id.items()}

df_reviews['user_id'] = df_reviews['name'].map(user2id)
df_reviews['item_id'] = df_reviews['asin'].map(item2id)

num_users = len(user2id)
num_items = len(item2id)
print(f"  Matrix Shape  : {num_users} users × {num_items} items")

# ── 3. Build Sparse Matrix & SVD ─────────────────────────────────────────────
print("[3/5] Building sparse matrix and running SVD...")
# We aggregate multiple reviews by the same user for the same item by taking the max rating
df_agg = df_reviews.groupby(['user_id', 'item_id'])['rating'].max().reset_index()

ratings_matrix = csr_matrix(
    (df_agg['rating'], (df_agg['user_id'], df_agg['item_id'])),
    shape=(num_users, num_items)
)

# Normalize by subtracting user mean rating (to handle optimistic/pessimistic raters)
user_ratings_mean = np.array(ratings_matrix.mean(axis=1))
ratings_diff = ratings_matrix.copy()

# Fast SVD (rank 50 or min dimension - 1)
k = min(50, num_items - 1)
U, sigma, Vt = svds(ratings_matrix.astype(float), k=k)

# Sort descending
sigma = np.diag(sigma)[::-1, ::-1]
U = U[:, ::-1]
Vt = Vt[::-1, :]

print(f"  SVD computed with rank {k}.")

# ── 4. Build Product Catalog ─────────────────────────────────────────────────
print("[4/5] Building product catalog...")
catalog = []
for item_id in range(num_items):
    asin = id2item[item_id]
    row = df_items.loc[asin]
    price = row.get('price', 0)
    if pd.isna(price) or price == 0:
        price = 99.99 # Fallback price
        
    catalog.append({
        "id": f"P-{item_id:04d}",
        "asin": asin,
        "name": str(row.get('title', f"Product {asin}")),
        "brand": str(row.get('brand', 'Amazon')),
        "category": "Cell Phones & Accessories",
        "price": float(price),
        "rating": float(row.get('rating', 4.0)),
        "total_reviews": int(row.get('totalReviews', 0)),
        "image_url": str(row.get('image', '')),
        "internal_id": item_id
    })

# ── 5. Save Artifacts ────────────────────────────────────────────────────────
print("[5/5] Saving model artifacts...")
with open(f"{OUT_DIR}/svd_U.npy", "wb") as f:
    np.save(f, U)
with open(f"{OUT_DIR}/svd_sigma.npy", "wb") as f:
    np.save(f, sigma)
with open(f"{OUT_DIR}/svd_Vt.npy", "wb") as f:
    np.save(f, Vt)
with open(f"{OUT_DIR}/user_ratings_mean.npy", "wb") as f:
    np.save(f, user_ratings_mean)
    
with open(f"{OUT_DIR}/user2id.json", "w") as f:
    json.dump(user2id, f)
with open(f"{OUT_DIR}/item2id.json", "w") as f:
    json.dump(item2id, f)
with open(f"{OUT_DIR}/catalog.json", "w") as f:
    json.dump(catalog, f)

metrics = {
    "num_users": num_users,
    "num_items": num_items,
    "num_interactions": len(df_agg),
    "sparsity": round(1.0 - (len(df_agg) / (num_users * num_items)), 6),
    "svd_rank": k,
    "dataset": "Amazon Cell Phones Reviews (Kaggle)"
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Recommendation model trained and saved to {OUT_DIR}/")
print(json.dumps(metrics, indent=2))
print("Training complete.")
