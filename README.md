# 📊 E-commerce Churn Intelligence

End-to-end machine learning pipeline for predicting customer churn, estimating customer value, and prioritizing retention actions.

---

## 🚀 Project Overview

This project builds a complete churn intelligence system using transactional data.

The goal is not only to predict churn, but to identify high-value customers at risk and recommend retention actions.

---

## 🧠 Problem Statement

Customer churn is a major challenge in e-commerce.

This project answers:

- Which customers are likely to churn?
- Which customers are valuable?
- Who should be targeted first?
- What action should be taken?

---

## ⚙️ Pipeline Architecture

```text
Raw Data → Cleaning → Features → Churn Model → CLV Model → Decision Layer → Strategy Layer
```

---

## 📈 Model Performance

```text
ROC AUC: ~0.72
Accuracy: ~0.65
Precision: ~0.59
Recall: ~0.60
```

Leakage-free snapshot approach ensures realistic results.

---

## 📊 Feature Importance

![Feature Importance](outputs/feature_importance.png)

Key drivers of churn:

- Customer value metrics
- Recency (last purchase)
- Activity trends

---

## 💰 CLV-Based Retention Priority

```text
priority_score = churn_probability × predicted_clv
```

This identifies customers who are both:

- likely to churn
- valuable

---

## 📊 Value vs Risk Matrix

![Value Risk Matrix](outputs/value_risk_matrix.png)

Customers segmented by:

- Value (CLV)
- Risk (churn)

### Interpretation

- High Value + High Risk → target immediately  
- High Value + Low Risk → nurture  
- Low Value + High Risk → low priority  

---

## 🎯 Retention Strategy Layer

The system recommends actions based on customer segment.

| Segment | Recommended Action |
|--------|------------------|
| HIGH_VALUE_HIGH_RISK | offer_discount |
| HIGH_VALUE_MEDIUM_RISK | personal_offer |
| HIGH_VALUE_LOW_RISK | loyalty_program |
| MEDIUM_VALUE_HIGH_RISK | email_campaign |
| LOW_VALUE_HIGH_RISK | low_priority |
| Other | no_action |

### Example

| CustomerID | Segment | Action |
|---|---|---|
| 16532 | HIGH_VALUE_HIGH_RISK | offer_discount |
| 12435 | HIGH_VALUE_MEDIUM_RISK | personal_offer |
| 12409 | HIGH_VALUE_LOW_RISK | loyalty_program |

Each recommendation includes a business explanation for better decision transparency.
---

## 📁 Project Structure

```text
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
│   ├── clv.py
│   ├── decisioning.py
│   ├── strategy.py
│   └── plots.py
│
├── outputs/
│   ├── customer_features.csv
│   ├── churn_priority_table.csv
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── value_risk_matrix.png
│   └── model_metrics.json
│
└── README.md
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python -m src.run_pipeline
```

---

## 📦 Outputs

- customer_features.csv  
- churn_priority_table.csv  
- feature_importance.csv  
- feature_importance.png  
- value_risk_matrix.png  
- model_metrics.json  

---

## 💡 Business Value

- Identify churn risk  
- Estimate customer value  
- Prioritize retention  
- Recommend actions  

---

## 🔗 Related Project

Customer Lifetime Value (CLV) project:

https://github.com/Coltrane35/customer-lifetime-value-retail

---

## 🔮 Future Improvements

- Campaign ROI simulation  
- Dashboard (Streamlit)  
- Advanced models  

---

## 👨‍💻 Author

Grzegorz Rączka

---

## ⭐ Key Takeaway

❌ churn prediction  

→  

✅ churn + CLV + decision + strategy system