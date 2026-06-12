import os
import joblib
import pandas as pd
import numpy as np

# ==========================
# Paths
# ==========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "train.csv"
)

MODELS = {
    "SVD": os.path.join(
        BASE_DIR,
        "models",
        "svd.pkl"
    ),
    "UserCF": os.path.join(
        BASE_DIR,
        "models",
        "usercf.pkl"
    ),
    "ItemCF": os.path.join(
        BASE_DIR,
        "models",
        "itemcf.pkl"
    ),
}

# ==========================
# Cache
# ==========================

_data_cache = None
_models_cache = {}

# ==========================
# Load Data
# ==========================

def load_data():

    global _data_cache

    if _data_cache is None:

        print(
            f"Loading data from {DATA_PATH} ..."
        )

        _data_cache = pd.read_csv(
            DATA_PATH
        )

        print(
            f"Loaded {len(_data_cache):,} ratings"
        )

    return _data_cache

# ==========================
# Load Model
# ==========================

def load_model(
    model_name="SVD"
):

    if model_name not in _models_cache:

        path = MODELS[model_name]

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"{model_name} not found."
            )

        print(
            f"Loading {model_name} model..."
        )

        _models_cache[model_name] = (
            joblib.load(path)
        )

    return _models_cache[
        model_name
    ]

# ==========================
# Recommendations
# ==========================

def recommend_movies(
    user_id: int,
    model_name: str = "SVD",
    top_n: int = 10,
    min_ratings: int = 20,
) -> pd.DataFrame:

    df = load_data()
    model = load_model(model_name)

    if user_id not in df["user_id"].values:
        raise ValueError(
            f"User ID {user_id} not found."
        )

    # =====================================
    # User Activity
    # =====================================

    user_history = df[
        df["user_id"] == user_id
    ]

    num_ratings = len(
        user_history
    )

    print("\n")
    print("=" * 50)
    print(f"USER ID: {user_id}")
    print(f"NUM RATINGS: {num_ratings}")
    print("=" * 50)

   

    if num_ratings < 10:

        pop_weight = 0.25

    elif num_ratings < 30:

        pop_weight = 0.10

    else:

        pop_weight = 0.05

    svd_weight = (
        1 - pop_weight
    )

    print(
        f"User ratings: {num_ratings}"
    )

    print(
        f"SVD weight: {svd_weight}"
    )

    print(
        f"Popularity weight: {pop_weight}"
    )

    # =====================================
    # Movie Statistics
    # =====================================

    movie_stats = (
        df.groupby("movie_id")
        .agg(
            avg_rating=("rating", "mean"),
            ratings_count=("rating", "count")
        )
        .reset_index()
    )

    movie_stats["popularity_score"] = (

        movie_stats["avg_rating"]

        *

        np.log1p(
            movie_stats["ratings_count"]
        )

    )

    movie_stats["popularity_score"] /= (
        movie_stats[
            "popularity_score"
        ].max()
    )

    popularity_dict = dict(
        zip(
            movie_stats["movie_id"],
            movie_stats["popularity_score"]
        )
    )

    movie_counts = dict(
        zip(
            movie_stats["movie_id"],
            movie_stats["ratings_count"]
        )
    )

    # =====================================
    # Watched Movies
    # =====================================

    watched = set(
        user_history[
            "movie_id"
        ]
    )

    # =====================================
    # Candidate Pool
    # =====================================

    candidates = list(

        set(

            movie_stats[
                movie_stats[
                    "ratings_count"
                ] >= min_ratings
            ]["movie_id"]

        )

        - watched

    )

    print(
        f"Candidates: {len(candidates)}"
    )

    if not candidates:

        raise ValueError(
            "No candidate movies found."
        )

    # =====================================
    # Stage 1:
    # Pure SVD Ranking
    # =====================================

    svd_predictions = []

    for movie_id in candidates:

        svd_score = model.predict(
            user_id,
            movie_id
        ).est

        svd_predictions.append(
            (
                movie_id,
                svd_score
            )
        )

    svd_predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    

    # =====================================
    # Stage 2:
    # Hybrid Re-ranking
    # =====================================

    predictions = []

    for movie_id in candidates:

        svd_score = model.predict(
            user_id,
            movie_id
        ).est

        pop_score = popularity_dict.get(
            movie_id,
            0
        )

        hybrid_score = (

            svd_weight
            * svd_score

            +

            pop_weight
            * (pop_score * 5)

        )

        predictions.append(

            (
                movie_id,
                hybrid_score,
                svd_score
            )

        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # =====================================
    # Metadata
    # =====================================

    meta_cols = [
        c
        for c in ["title", "year"]
        if c in df.columns
    ]

    if meta_cols:

        movie_meta = (

            df.drop_duplicates(
                "movie_id"
            )

            .set_index(
                "movie_id"
            )[meta_cols]

        )

    else:

        movie_meta = pd.DataFrame()

    results = []

    for rank, (
        movie_id,
        hybrid_score,
        svd_score
    ) in enumerate(

        predictions[:top_n],

        start=1

    ):

        if (
            not movie_meta.empty
            and movie_id
            in movie_meta.index
        ):

            row_meta = movie_meta.loc[
                movie_id
            ]

            title = row_meta[
                "title"
            ]

            year = row_meta[
                "year"
            ]

        else:

            title = (
                f"Movie {movie_id}"
            )

            year = "N/A"

        results.append({

            "rank":
            rank,

            "movie_id":
            movie_id,

            "title":
            title,

            "year":
            year,

            "predicted_rating":
            round(
                svd_score,
                3
            ),

            "hybrid_score":
            round(
                hybrid_score,
                3
            ),

            "ratings_count":
            int(
                movie_counts.get(
                    movie_id,
                    0
                )
            )

        })

    return pd.DataFrame(
        results
    )

# ==========================
# User History
# ==========================

def get_user_history(
    user_id,
    top_n=10
):

    df = load_data()

    history = (

        df[
            df["user_id"]
            == user_id
        ]

        .sort_values(
            "rating",
            ascending=False
        )

        .head(top_n)

    )

    return history[
        [
            "movie_id",
            "title",
            "year",
            "rating"
        ]
    ]

# ==========================
# Dataset Stats
# ==========================

def get_dataset_stats():

    df = load_data()

    return {
        "total_ratings": f"{len(df):,}",
        "unique_users": f"{df['user_id'].nunique():,}",
        "unique_movies": f"{df['movie_id'].nunique():,}",
        "avg_rating": round(df["rating"].mean(), 2),
    }

# ==========================
# CLI TESTING
# ==========================

if __name__ == "__main__":

    stats = (
        get_dataset_stats()
    )

    print(
        "\nDATASET STATS\n"
    )

    for k, v in stats.items():

        print(
            f"{k}: {v}"
        )

    df = load_data()

    heavy_user = (

        df[
            "user_id"
        ]

        .value_counts()

        .idxmax()

    )

    print(
        f"\nHEAVY USER: {heavy_user}"
    )

    print(
        "\nUSER HISTORY\n"
    )

    print(

        get_user_history(
            heavy_user
        )

    )

    print(
        "\nRECOMMENDATIONS\n"
    )

    recs = recommend_movies(
        user_id=heavy_user,
        model_name="SVD",
        top_n=10,
        min_ratings=20
    )

    print(recs)

    print("\nHEAVY USER RECS")
    print(
        recommend_movies(
            2439493,
            top_n=10
        )
    )
    print("\nCOLD USER RECS")
    print(
        recommend_movies(
            788,
            top_n=10
        )
    )
