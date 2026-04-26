# 📊 E-commerce Churn Intelligence

End-to-end machine learning pipeline for predicting customer churn and prioritizing retention actions in e-commerce.

---

## 🚀 Project Overview

This project builds a complete churn prediction system using transactional data from an online retail dataset.

The goal is not only to predict churn, but to **identify high-value customers at risk and support business decisions**.

---

## 🧠 Problem Statement

Customer churn is one of the biggest challenges in e-commerce.

Instead of simply predicting churn, this project answers:

- Which customers are likely to churn?
- Which of them are valuable?
- Who should be targeted first in retention campaigns?

---

## ⚙️ Pipeline Architecture


Raw Data → Cleaning → Feature Engineering → Churn Labeling → Model Training → Decision Layer


### Steps:

### 1. Data Cleaning
- Removed invalid transactions (negative quantity, missing customers)
- Correct timestamp parsing (robust handling of multiple formats)

### 2. Feature Engineering
- RFM features:
  - Recency
  - Frequency
  - Monetary value
- Time-window features:
  - Activity in last 30 / 60 / 90 days
- Trend indicators:
  - Behavioral change over time

### 3. Churn Labeling (NO DATA LEAKAGE)
- Snapshot-based approach:
  - Features built using historical data
  - Churn defined in a future window (90 days)
- Ensures realistic model performance

### 4. Modeling
- Logistic Regression (with scaling)
- CatBoost (baseline comparison)
- Automatic model selection based on ROC AUC

### 5. Decision Layer
- Customer churn scoring
- Priority ranking for retention campaigns

---

## 📈 Model Performance


ROC AUC: ~0.72
Accuracy: ~0.65
Precision: ~0.59
Recall: ~0.60


✔ Model is realistic (no data leakage)  
✔ Designed for business usage, not Kaggle overfitting  

---

## 📊 Feature Importance

![Feature Importance](outputs/feature_importance.png)

### 🔍 Interpretation

The model highlights several key drivers of churn:

- **Customer value (monetary_mean, monetary_median)**  
  → High-value customers behave differently and are easier to model

- **Recent activity (recency_days)**  
  → Time since last purchase is one of the strongest churn indicators

- **Recent spend (spend_last_60d, spend_last_30d)**  
  → Drop in spending signals disengagement

- **Behavior trends (trend_spend_30_vs_90)**  
  → Declining activity increases churn probability

### 💡 Business Insight

Customers who:
- haven't purchased recently  
- spend less than before  
- show declining activity  

are significantly more likely to churn.

---

## 📁 Project Structure

```
ecommerce-churn-intelligence/
│
├── data/
│   └── raw/
│       └── Online_Retail.csv
│
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── churn_label.py
│   ├── features.py
│   ├── modeling.py
│   ├── decisioning.py
│   └── plots.py
│
├── outputs/
│   ├── customer_features.csv
│   ├── churn_priority_table.csv
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   └── model_metrics.json
│
└── README.md
```


---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Run pipeline
python -m src.run_pipeline
3. Outputs

Generated files:

customer_features.csv → model-ready dataset
churn_priority_table.csv → prioritized retention targets
feature_importance.csv → explainability data
feature_importance.png → visual insights
model_metrics.json → performance metrics
💡 Business Value

This project provides:

Identification of customers at risk of churn
Prioritization for retention campaigns
Behavioral insights for marketing teams
Foundation for data-driven decision making
🔮 Future Improvements
Customer Lifetime Value (CLV) integration
ROI-based retention simulation
Streamlit dashboard
Advanced models (XGBoost / LightGBM)
Time-based modeling
👨‍💻 Author

Grzegorz Rączka
Machine Learning / Data Science

⭐ Key Takeaway

This project demonstrates how to move from:

❌ simple churn prediction

to:

✅ actionable churn intelligence system
➡️ focusing on real business impact, not just model performance