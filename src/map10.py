import pandas as pd
import numpy as np
import joblib

# ==========================
# Load Data
# ==========================

print("Loading datasets...")

train_df = pd.read_csv(
    "data/processed/train.csv"
)

test_df = pd.read_csv(
    "data/processed/test.csv"
)

# ==========================
# Load Trained SVD
# ==========================

print("Loading trained SVD...")

model = joblib.load(
    "models/svd.pkl"
)

# ==========================
# Active Users
# ==========================

user_counts = train_df["user_id"].value_counts()

active_users = user_counts[
    user_counts >= 15
].index

print(
    "Users evaluated:",
    len(active_users)
)

# ==========================
# Relevant Movies
# ==========================

relevant_movies = (
    test_df[
        test_df["rating"] >= 3.5
    ]
    .groupby("user_id")
    ["movie_id"]
    .apply(set)
    .to_dict()
)

# ==========================
# Popularity Scores
# ==========================

print("Calculating popularity scores...")

movie_stats = (
    train_df
    .groupby("movie_id")
    .agg({
        "rating": "mean",
        "user_id": "count"
    })
    .reset_index()
)

movie_stats.columns = [
    "movie_id",
    "avg_rating",
    "ratings_count"
]

movie_stats["popularity_score"] = (

    movie_stats["avg_rating"]

    *

    np.log1p(
        movie_stats["ratings_count"]
    )

)

movie_stats["popularity_score"] /= (

    movie_stats["popularity_score"]
    .max()

)

popularity_dict = dict(

    zip(
        movie_stats["movie_id"],
        movie_stats["popularity_score"]
    )

)

# ==========================
# Candidate Pools
# ==========================

movie_counts = (
    train_df
    .groupby("movie_id")
    .size()
)

candidate_pools = {

    "min_20": set(
        movie_counts[
            movie_counts >= 20
        ].index
    ),

    "min_50": set(
        movie_counts[
            movie_counts >= 50
        ].index
    ),

    "min_100": set(
        movie_counts[
            movie_counts >= 100
        ].index
    ),

}

for name, pool in candidate_pools.items():

    print(
        f"Candidate pool [{name}]: {len(pool):,} movies"
    )

# ==========================
# AP@10
# ==========================

def average_precision_at_k(
    actual,
    predicted,
    k=10
):

    predicted = predicted[:k]

    score = 0
    hits = 0

    for i, movie in enumerate(predicted):

        if movie in actual:

            hits += 1

            score += (
                hits /
                (i + 1)
            )

    if len(actual) == 0:

        return 0

    return (
        score /
        min(len(actual), k)
    )

# ==========================
# MAP@10 Evaluation
# ==========================

results = {}

for pool_name, candidate_pool in candidate_pools.items():

    print(
        f"\nRunning MAP@10 — {pool_name}"
    )

    ap_scores = []

    skipped_no_relevant = 0

    skipped_relevant_filtered = 0

    for idx, user_id in enumerate(
        active_users
    ):

        if idx % 100 == 0:

            print(
                f"Processed {idx}/{len(active_users)} users"
            )

        actual = relevant_movies.get(
            user_id,
            set()
        )

        if len(actual) == 0:

            skipped_no_relevant += 1

            continue

        watched = set(

            train_df[
                train_df["user_id"]
                == user_id
            ]["movie_id"]

        )

        candidate_movies = (
            candidate_pool
            - watched
        )

        relevant_in_candidates = (
            actual
            &
            candidate_movies
        )

        if len(
            relevant_in_candidates
        ) == 0:

            skipped_relevant_filtered += 1

            continue

        # ==================
        # Dynamic Weighting
        # ==================

        num_ratings = len(

            train_df[
                train_df["user_id"]
                == user_id
            ]

        )

        if num_ratings < 10:

            pop_weight = 0.25

        elif num_ratings < 30:

            pop_weight = 0.15

        else:

            pop_weight = 0.05

        svd_weight = (
            1 - pop_weight
        )

        predictions = []

        for movie_id in candidate_movies:

            svd_score = (
                model.predict(
                    user_id,
                    movie_id
                ).est
            )

            pop_score = (
                popularity_dict.get(
                    movie_id,
                    0
                )
            )

            final_score = (

                svd_weight
                * svd_score

                +

                pop_weight
                * (pop_score * 5)

            )

            predictions.append(
                (
                    movie_id,
                    final_score
                )
            )

        predictions.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top10 = [

            movie

            for movie, score

            in predictions[:10]

        ]

        ap = average_precision_at_k(
            actual,
            top10,
            k=10
        )

        ap_scores.append(ap)

    map10 = (

        sum(ap_scores)

        /

        len(ap_scores)

        if ap_scores

        else 0

    )

    results[pool_name] = {

        "map10":
        round(map10, 4),

        "users_evaluated":
        len(ap_scores),

        "skipped_relevant_filtered":
        skipped_relevant_filtered,

        "candidate_pool_size":
        len(candidate_pool),

    }

# ==========================
# Final Results
# ==========================

print("\n")
print("=" * 70)
print("FINAL MAP@10 RESULTS")
print("=" * 70)

print(
    f"{'Pool':<12}"
    f"{'Candidates':>12}"
    f"{'Users Eval':>14}"
    f"{'Skipped':>12}"
    f"{'MAP@10':>10}"
)

print("-" * 70)

for pool_name, r in results.items():

    print(

        f"{pool_name:<12}"

        f"{r['candidate_pool_size']:>12,}"

        f"{r['users_evaluated']:>14,}"

        f"{r['skipped_relevant_filtered']:>12,}"

        f"{r['map10']:>10.4f}"

    )

print("=" * 70)