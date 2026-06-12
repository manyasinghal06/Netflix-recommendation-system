import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
from recommend import (
    recommend_movies,
    get_user_history,
    get_dataset_stats,
    load_data,
    load_model,
)

# ==========================
# Page config
# ==========================

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
)

# ==========================
# CSS overrides
# ==========================

st.markdown("""
<style>
/* Dark background for the whole app */
[data-testid="stAppViewContainer"] {
    background-color: #0f0f0f;
}
[data-testid="stHeader"] {
    background-color: #0f0f0f;
}
[data-testid="stSidebar"] {
    background-color: #161616;
}

/* General text */
html, body, [class*="css"] {
    color: #e0e0e0;
}

/* Markdown headings & text */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #e0e0e0;
}

/* Streamlit caption */
[data-testid="stCaptionContainer"] p {
    color: #888 !important;
}

/* Inputs */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #1e1e1e !important;
    color: #e0e0e0 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input:disabled {
    background-color: #1a1a1a !important;
    color: #e0e0e0 !important;
    -webkit-text-fill-color: #e0e0e0 !important;
    opacity: 1 !important;
}

/* Slider */
[data-testid="stSlider"] {
    color: #e0e0e0;
}

/* Button */
[data-testid="stButton"] button[kind="primary"] {
    background-color: #e50914;
    border: none;
    color: #fff;
    font-weight: 600;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #c40812;
}

/* Download button */
[data-testid="stDownloadButton"] button {
    background-color: #1e1e1e;
    border: 1px solid #333;
    color: #e0e0e0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #1a1a1a;
}
.stDataFrame thead tr th {
    background-color: #222 !important;
    color: #e0e0e0 !important;
}
.stDataFrame tbody tr td {
    background-color: #1a1a1a !important;
    color: #e0e0e0 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background-color: #1e1e1e;
    border-color: #333;
    color: #e0e0e0;
}

/* Metric cards */
.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-label {
    font-size: 12px;
    color: #666;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 22px;
    font-weight: 600;
    color: #e0e0e0;
}

/* Movie cards */
.movie-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 8px;
}
.rank-badge {
    font-size: 11px;
    font-weight: 600;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.movie-title {
    font-size: 15px;
    font-weight: 600;
    color: #f0f0f0;
    margin: 2px 0;
}
.movie-meta {
    font-size: 13px;
    color: #666;
}
.rating-pill {
    display: inline-block;
    background: #1b3a1f;
    color: #4caf50;
    font-size: 13px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
}

/* RMSE bar */
.rmse-bar-wrap {
    background: #2a2a2a;
    border-radius: 6px;
    height: 8px;
    margin-top: 6px;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 1.5rem 0;
}

/* Selectbox / dropdowns */
[data-testid="stSelectbox"] > div > div {
    background-color: #1e1e1e !important;
    color: #e0e0e0 !important;
    border: 1px solid #333 !important;
}

/* Spinner text */
[data-testid="stSpinner"] p {
    color: #888 !important;
}

/* Table in markdown */
table {
    background-color: #1a1a1a;
    color: #e0e0e0;
    border-color: #2a2a2a;
}
th {
    background-color: #222 !important;
    color: #e0e0e0 !important;
    border-color: #333 !important;
}
td {
    border-color: #2a2a2a !important;
    color: #e0e0e0 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Cached loaders
# ==========================

@st.cache_data
def cached_stats():
    return get_dataset_stats()

@st.cache_data
def cached_valid_users():
    df = load_data()
    return set(df["user_id"].unique())

@st.cache_resource
def cached_model(name):
    return load_model(name)

# ==========================
# Header
# ==========================

st.markdown("## 🎬 Netflix recommendation system")
st.caption(
    "Netflix Prize Dataset · Dynamic Hybrid SVD Recommender"
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ==========================
# Stats bar
# ==========================

stats = cached_stats()

model_rmse = {
    "SVD"   : ("0.961", "#2e7d32"),
    "UserCF": ("1.028", "#888"),
    "ItemCF": ("1.084", "#888"),
}

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total ratings</div>
        <div class="metric-value">{stats['total_ratings']}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Users</div>
        <div class="metric-value">{stats['unique_users']}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Movies</div>
        <div class="metric-value">{stats['unique_movies']}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg rating</div>
        <div class="metric-value">{stats['avg_rating']}</div>
    </div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best RMSE (SVD)</div>
        <div class="metric-value" style="color:#4caf50">0.961</div>
    </div>""", unsafe_allow_html=True)
with c6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MAP@10 (Hybrid)</div>
        <div class="metric-value" style="color:#42a5f5">0.0028</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ==========================
# Main layout — 2 columns
# ==========================

left, right = st.columns([1, 2], gap="large")

with left:

    st.markdown("#### Controls")

    valid_users = cached_valid_users()

    user_id = st.number_input(
        "User ID",
        min_value=1,
        step=1,
        help="Enter a user ID from the training dataset",
    )

    st.text_input(
        "Recommendation model",
        value="Dynamic Hybrid SVD",
        disabled=True,
    )

    model_name = "Dynamic Hybrid SVD"

    top_n = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
    )

    min_ratings = st.slider(
        "Min movie popularity (ratings count)",
        min_value=20,
        max_value=100,
        value=20,
        help="Only recommend movies with at least this many ratings in the dataset",
    )

    recommend_btn = st.button(
        "Get recommendations",
        type="primary",
        use_container_width=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("#### Model performance")

    rmse_vals = {"SVD": 0.961, "UserCF": 1.028, "ItemCF": 1.084}
    rmse_max  = 1.084
    rmse_min  = 0.960

    for name, rmse in rmse_vals.items():
        is_active = name == "SVD"
        bar_pct   = int((1 - (rmse - rmse_min) / (rmse_max - rmse_min + 0.001)) * 100)
        color     = "#4caf50" if name == "SVD" else "#42a5f5" if name == "UserCF" else "#888"
        weight    = "700" if is_active else "400"
        st.markdown(f"""
        <div style="margin-bottom:10px">
            <div style="display:flex;justify-content:space-between;font-size:13px">
                <span style="font-weight:{weight};color:#e0e0e0">{name} {'←' if is_active else ''}</span>
                <span style="color:{color};font-weight:600">RMSE {rmse}</span>
            </div>
            <div class="rmse-bar-wrap">
                <div style="width:{bar_pct}%;background:{color};height:8px;border-radius:6px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.caption("Lower RMSE = better rating prediction accuracy")

with right:

    if recommend_btn:

        if int(user_id) not in valid_users:
            st.error(
                f"User ID **{int(user_id)}** not found in training data. "
                f"Try a different ID."
            )
            sample_ids = list(valid_users)[:5]
            st.info(f"Sample valid user IDs: {sample_ids}")

        else:
            uid = int(user_id)

            # --- user history ---
            with st.spinner("Loading user history..."):
                try:
                    history = get_user_history(uid, top_n=5)
                    st.markdown("##### Movies this user has rated")
                    st.dataframe(
                        history.rename(columns={
                            "title": "Title",
                            "year" : "Year",
                            "rating": "Rating ⭐",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as e:
                    st.warning(f"Could not load user history: {e}")

            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

            # --- recommendations ---
            st.markdown(f"##### Top {top_n} recommendations · {model_name}")

            with st.spinner(f"Running {model_name}..."):
                try:
                    recs = recommend_movies(
                        user_id     = uid,
                        top_n       = top_n,
                        min_ratings = min_ratings,
                    )

                    # movie cards
                    for _, row in recs.iterrows():
                        score = row.get(
                            "hybrid_score",
                            row["predicted_rating"]
                        )

                        stars = "⭐" * int(round(score))
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="rank-badge">#{int(row['rank'])}</div>
                            <div class="movie-title">{row['title']}</div>
                            <div class="movie-meta">
                                {row['year']} &nbsp;·&nbsp;
                                {int(row['ratings_count'])} ratings in dataset
                            </div>
                            <div style="margin-top:6px">
                                <span class="rating-pill">★ {score}</span>
                                &nbsp;<span style="font-size:12px;color:#555">{stars}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                    # download button
                    csv = recs.to_csv(index=False)
                    st.download_button(
                        label     = "Download recommendations as CSV",
                        data      = csv,
                        file_name = f"recommendations_user{uid}_{model_name}.csv",
                        mime      = "text/csv",
                    )

                except FileNotFoundError as e:
                    st.error(f"Model not loaded: {e}")
                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    else:
        st.markdown("#### How it works")
        st.markdown("""
        1. Enter a **User ID** from the Netflix Prize dataset
        2. Adjust **Top-K recommendations**
        3. Adjust **minimum popularity threshold**
        4. Click **Get recommendations**

        ---

        ### Model Comparison

        | Model | RMSE |
        |--------|--------:|
        | SVD | 0.961 |
        | UserCF | 1.028 |
        | ItemCF | 1.084 |

        ---

        ### Final Production Model

        **Dynamic Hybrid SVD**

        - Matrix Factorization (SVD)
        - Dynamic popularity weighting
        - Heavy users: 95% SVD + 5% popularity
        - Medium users: 90% SVD + 10% popularity
        - Cold users: 75% SVD + 25% popularity

        Movies already watched are excluded before ranking.

        Popularity is used only to stabilize recommendations for sparse users while preserving personalization for active users.
        """)