# 📉 Telco Customer Churn Prediction Pipeline

An end-to-end machine learning project that predicts customer churn in the telecommunications sector. This repository contains the full workflow from exploratory data analysis to a deployable production training script.

## 🚀 Business Context
Customer retention is a critical KPI for telecom companies. Acquiring a new customer is significantly more expensive than retaining an existing one.

* **Problem:** Identifying which customers are at high risk of leaving (churning) before they actually leave.
* **Goal:** Build a model that prioritizes **Recall** (catching as many churners as possible) to enable proactive retention campaigns.
* **Constraint:** Minimize false negatives (missed churners) while maintaining a reasonable precision to avoid wasting marketing budget.

## 📊 Key Results
The final model optimizes for **Recall** to ensure high-risk customers are identified.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.83** | Strong ability to rank customers by risk probability. |
| **Recall (Churn)** | **~75%** | The model correctly identifies 3 out of every 4 churners. |
| **Precision** | **~54%** | Acceptable trade-off to ensure maximum churner capture. |

## 🛠️ Tech Stack & Techniques
* **Python 3.12**
* **XGBoost:** Gradient boosting classifier for high-performance tabular prediction.
* **Scikit-Learn:** Pipeline architecture, preprocessing, and evaluation.
* **Imbalanced Learning:** Implemented `scale_pos_weight` to handle the class imbalance (26% Churn vs 74% Stay).
* **Hyperparameter Tuning:** RandomizedSearchCV to optimize `learning_rate`, `max_depth`, and tree parameters.
* **Threshold Tuning:** Custom probability thresholding to maximize Recall.

## 📂 Repository Structure

```text
├── telco-churn-analysis.ipynb   # 📓 Jupyter Notebook: EDA, experiments, and visualization
├── train.py                     # ⚙️ Production Script: Cleans data, trains model, saves artifact
├── requirements.txt             # 📦 Dependencies: List of libraries for reproducibility
├── churn_model.joblib           # 🧠 Saved Model: Serialized pipeline ready for inference
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # 💾 Dataset (Source: Kaggle)
```

## ⚡ Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/shank187/customer-churn-prediction-pipeline
   cd customer-churn-prediction-pipeline
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**

   Run the production script to train the model and save it to `churn_model.joblib`:

   ```bash
   python train.py
   ```

## 📈 Key Insights

* **Contract Type:** Customers on **Month-to-Month** contracts are the highest churn risk.
* **Internet Service:** **Fiber Optic** users churn significantly more than DSL users, indicating potential service or pricing dissatisfaction.
* **Payment Method:** **Electronic Check** payers have a higher tendency to leave compared to automatic payment users.
