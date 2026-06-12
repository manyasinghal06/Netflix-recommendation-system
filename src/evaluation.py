import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import KNNWithMeans
from surprise import accuracy

# ==========================
# Load Official Split
# ==========================

print("Loading datasets...")

train_df = pd.read_csv(
    "data/processed/train.csv"
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)

print("Train Shape:", train_df.shape)
print("Test Shape :", test_df.shape)

# ==========================
# Build Surprise Trainset
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

# Surprise test format:
# (user, item, true_rating)

testset = list(
    zip(
        test_df["user_id"],
        test_df["movie_id"],
        test_df["rating"]
    )
)

results = []

# =====================================================
# USER CF
# =====================================================

print("\n===== USER CF =====")

user_counts = train_df["user_id"].value_counts()

active_users = user_counts[
    user_counts >= 10
].index

train_usercf = train_df[
    train_df["user_id"].isin(active_users)
]

test_usercf = test_df[
    test_df["user_id"].isin(active_users)
]

train_data_usercf = Dataset.load_from_df(
    train_usercf[
        ["user_id", "movie_id", "rating"]
    ],
    reader
)

trainset_usercf = train_data_usercf.build_full_trainset()

testset_usercf = list(
    zip(
        test_usercf["user_id"],
        test_usercf["movie_id"],
        test_usercf["rating"]
    )
)

user_model = KNNWithMeans(
    k=40,
    min_k=3,
    sim_options={
        "name": "cosine",
        "user_based": True
    },
    verbose=True
)

user_model.fit(trainset_usercf)

predictions = user_model.test(
    testset_usercf
)

user_rmse = accuracy.rmse(
    predictions,
    verbose=True
)

results.append(
    ["UserCF", user_rmse]
)

# =====================================================
# ITEM CF
# =====================================================

print("\n===== ITEM CF =====")

item_model = KNNWithMeans(
    k=40,
    min_k=3,
    sim_options={
        "name": "cosine",
        "user_based": False
    },
    verbose=True
)

item_model.fit(trainset_usercf)

predictions = item_model.test(
    testset_usercf
)

item_rmse = accuracy.rmse(
    predictions,
    verbose=True
)

results.append(
    ["ItemCF", item_rmse]
)

# =====================================================
# SVD
# =====================================================

print("\n===== SVD =====")

svd_model = SVD(
    n_factors=100,
    n_epochs=20,
    random_state=42
)

svd_model.fit(trainset)

predictions = svd_model.test(
    testset
)

svd_rmse = accuracy.rmse(
    predictions,
    verbose=True
)

results.append(
    ["SVD", svd_rmse]
)

# =====================================================
# FINAL TABLE
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "RMSE"
    ]
)

print("\n========================")
print("FINAL RESULTS")
print("========================")

print(
    results_df.sort_values("RMSE")
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nResults saved!")