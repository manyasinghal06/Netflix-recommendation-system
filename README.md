# Netflix-recommendation-system
# 🎬 Netflix Recommendation System
### Dynamic Hybrid SVD Recommender using the Netflix Prize Dataset

A personalized movie recommendation system built using the Netflix Prize dataset. The project compares multiple collaborative filtering approaches and proposes a **Dynamic Hybrid SVD model** that adapts recommendation weights based on user activity levels to improve recommendation quality, especially for cold-start users.

---

## 🚀 Features

- User-Based Collaborative Filtering (UserCF)
- Item-Based Collaborative Filtering (ItemCF)
- Matrix Factorization using SVD
- Dynamic Hybrid SVD Recommendation Engine
- Cold-Start User Handling
- RMSE and MAP@10 Evaluation
- Interactive Streamlit Dashboard
- Top-K Personalized Movie Recommendations

---

## 📊 Dataset

Netflix Prize Dataset

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Ratings | 999,994 |
| Users | 289,387 |
| Movies | 16,144 |
| Sparsity | 99.98% |

---

## 🧠 Recommendation Approaches

### 1. User-Based Collaborative Filtering
Recommends movies using preferences of similar users.

### 2. Item-Based Collaborative Filtering
Recommends movies similar to items previously liked by a user.

### 3. Singular Value Decomposition (SVD)
Learns latent user and movie factors through matrix factorization.

### 4. Dynamic Hybrid SVD (Final Model)
Combines SVD predictions with popularity signals using adaptive weights based on user activity.

| User Type | SVD Weight | Popularity Weight |
|------------|------------|------------------|
| Cold User (<10 ratings) | 75% | 25% |
| Moderate User (<30 ratings) | 90% | 10% |
| Active User (≥30 ratings) | 95% | 5% |

Hybrid Score:

```text
Hybrid Score = (w_svd × SVD Score)
             + (w_pop × Popularity Score)
```

---

## 📈 Model Performance

### RMSE Comparison

| Model | RMSE |
|---------|---------|
| UserCF | YOUR_VALUE |
| ItemCF | YOUR_VALUE |
| SVD | 0.961 |

### MAP@10 Evaluation

| Candidate Pool | MAP@10 |
|----------------|---------|
| Min 20 Ratings | YOUR_VALUE |
| Min 50 Ratings | YOUR_VALUE |
| Min 100 Ratings | YOUR_VALUE |

---

## 🖥️ Dashboard Preview

### Home Dashboard

<img width="1867" height="852" alt="image" src="https://github.com/user-attachments/assets/89384a58-1831-41ce-bb2d-dc04e520c175" />
<img width="1541" height="763" alt="image" src="https://github.com/user-attachments/assets/57a8f37e-6cd6-4b3e-8691-e0ff211d3a61" />



### Recommendation Example

<img width="1841" height="800" alt="image" src="https://github.com/user-attachments/assets/7bc6a174-23b8-4049-be22-c5f28af6cce2" />

<img width="1866" height="808" alt="image" src="https://github.com/user-attachments/assets/14a2c76b-b457-41a1-8568-c30cc90e937d" />


## 📂 Project Structure

```text
Netflix-Recommendation-System/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── usercf.pkl
│   ├── itemcf.pkl
│   └── svd.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── results/
│   └── model_comparison.csv
│
├── report/
│   ├── technical_report.pdf
│   └── presentation.pdf
│
├── src/
│   ├── preprocessing.py
│   ├── build_model_dataset.py
│   ├── train_test_split.py
│   ├── user_cf.py
│   ├── item_cf.py
│   ├── svd.py
│   ├── evaluation.py
│   ├── map10.py
│   └── recommend.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd Netflix-Recommendation-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Reproducing Results

### Step 1: Data Preprocessing

```bash
python src/preprocessing.py
```

### Step 2: Build Modeling Dataset

```bash
python src/build_model_dataset.py
```

### Step 3: Train-Test Split

```bash
python src/train_test_split.py
```

### Step 4: Train Models

```bash
python src/user_cf.py

python src/item_cf.py

python src/svd.py
```

### Step 5: Evaluate Models

```bash
python src/evaluation.py

python src/map10.py
```

### Step 6: Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## 🎯 Key Contributions

- Compared UserCF, ItemCF, and SVD recommendation approaches.
- Developed a Dynamic Hybrid SVD model.
- Addressed cold-start user challenges through adaptive weighting.
- Evaluated recommendation quality using RMSE and MAP@10.
- Built an interactive Streamlit recommendation dashboard.

---

## 🔮 Future Improvements

- Content-Based Filtering
- Deep Learning Recommenders
- Genre-Aware Recommendations
- Real-Time Recommendation Updates

---

