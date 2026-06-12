import pandas as pd
import pickle

from surprise import Dataset
from surprise import Reader
from surprise import SVD

print("Loading training data...")

train_df = pd.read_csv(
    "data/processed/train.csv"
)

reader = Reader(rating_scale=(1, 5))

train_data = Dataset.load_from_df(
    train_df[
        ["user_id", "movie_id", "rating"]
    ],
    reader
)

trainset = train_data.build_full_trainset()

model = SVD(
    n_factors=200,
    n_epochs=50,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)
print("Training SVD...")
print(
    "n_factors=", model.n_factors,
    "epochs=", model.n_epochs
)

model.fit(trainset)

with open(
    "models/svd.pkl",
    "wb"
) as f:
    pickle.dump(model, f)

print("Model saved!")