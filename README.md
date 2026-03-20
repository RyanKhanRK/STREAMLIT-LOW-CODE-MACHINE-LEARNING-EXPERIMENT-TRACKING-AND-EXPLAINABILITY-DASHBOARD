# 🧪 Low-Code ML Experiment Tracking & Explainability Dashboard

A fully self-contained Streamlit application for end-to-end machine learning
experimentation — upload data, preprocess, train, compare, and explain — all
from a browser with zero code.

---

## ✨ Features

| Page | What you can do |
|------|----------------|
| 📂 **Data Upload** | Upload any CSV, auto-analyse structure, select target + features |
| ⚙️ **Preprocessing** | Drop/fill NaN, encode categoricals, scale features, live preview |
| 🤖 **Train & Evaluate** | Single model or batch compare; dynamic hyperparameter sliders; confusion matrix, scatter, feature importances |
| 📊 **Experiment Tracker** | Filter/compare all runs; radar chart; head-to-head diff; CSV export |
| 🔍 **Explainability** | SHAP summary, bar, waterfall, and dependence plots; feature importance table |

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Start MLflow tracking server
mlflow server --host 127.0.0.1 --port 5000

# 4. Launch the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
ml_dashboard/
├── app.py                     Main Streamlit entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml            Dark theme + server settings
├── modules/
│   ├── session.py             Session-state schema
│   ├── ui_styles.py           Custom CSS injection
│   ├── data_utils.py          CSV loading, analysis, preprocessing
│   ├── models.py              Model registry + training pipeline
│   ├── mlflow_utils.py        MLflow logging wrapper
│   └── explainability.py      SHAP helpers
└── pages/
    ├── data_upload.py
    ├── preprocessing.py
    ├── model_train.py
    ├── experiment_tracker.py
    └── explainability.py
```

---

## 🤖 Supported Models

**Classification:** Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, SVM, K-Nearest Neighbours, Naive Bayes

**Regression:** Linear, Ridge, Lasso, Decision Tree, Random Forest,
Gradient Boosting, SVR

**Clustering:** K-Means, DBSCAN, Agglomerative

---

## 📊 Metrics Tracked

| Task | Metrics |
|------|---------|
| Classification | Accuracy, F1 (weighted), Precision, Recall, ROC-AUC, CV mean/std |
| Regression | R², RMSE, MAE, CV R² mean/std |
| Clustering | Silhouette Score, Davies-Bouldin Score, # clusters found |

---

## 🔧 MLflow Integration

- Logs parameters, metrics, and model artifacts automatically.
- By default uses a local `mlruns/` directory (no server needed).
- Point to a remote server by entering its URL in the Train page.
- Disable logging by unchecking "Log to MLflow" on the Train page.

---

## 🔍 SHAP Explainability

- **TreeExplainer** — used automatically for Random Forest, Gradient Boosting, Decision Tree (fast, exact).
- **LinearExplainer** — for Logistic Regression, Ridge, Lasso (fast).
- **KernelExplainer** — fallback for SVM, KNN, etc. (slower; reduce sample size for speed).

---

## 📋 Supported Dataset Formats

- CSV or TSV with a header row.
- Any mix of numeric and categorical columns.
- Missing values handled in the Preprocessing step.

---

## 🔮 Future Improvements

- Hyperparameter search (Grid / Random / Optuna)
- MLflow Model Registry integration
- Docker + docker-compose deployment
- Automated fairness/bias metrics (`fairlearn`)
- PDF/HTML experiment report export
- Multi-user support with authentication
