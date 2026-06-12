import os
import pickle
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans

# ==========================
# Load Training Data
# ==========================

print("Loading training data...")

train_df = pd.read_csv(
    "data/processed/train.csv"
)

print("Original Shape:", train_df.shape)

# ==========================
# Active User Filtering
# ==========================

user_counts = train_df["user_id"].value_counts()

active_users = user_counts[
    user_counts >= 10
].index

train_df = train_df[
    train_df["user_id"].isin(active_users)
]

print("Filtered Shape:", train_df.shape)

# ==========================
# Convert To Surprise Format
# ==========================

reader = Reader(
    rating_scale=(1, 5)
)

train_data = Dataset.load_from_df(
    train_df[
        ["user_id", "movie_id", "rating"]
    ],
    reader
)

trainset = train_data.build_full_trainset()

# ==========================
# Item-Based CF
# ==========================

sim_options = {
    "name": "cosine",
    "user_based": False
}

model = KNNWithMeans(
    k=40,
    min_k=3,
    sim_options=sim_options,
    verbose=True
)

print("\nTraining ItemCF...")

model.fit(trainset)

# ==========================
# Save Model
# ==========================

os.makedirs(
    "models",
    exist_ok=True
)

with open(
    "models/itemcf.pkl",
    "wb"
) as f:
    pickle.dump(model, f)

print("\nItemCF model saved!")