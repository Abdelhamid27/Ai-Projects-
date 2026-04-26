# Bank Customer Churn Prediction 🚀

## 📌 Project Overview
This project aims to predict **Customer Churn** for a banking institution. By analyzing customer demographics and financial behavior, we developed a machine learning system that identifies customers who are likely to leave the bank.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Plotly.
* **Environment:** Jupyter Notebook.

## ⚙️ Data Preprocessing
1. **Encoding:** Applied **One-Hot Encoding** to categorical variables.
2. **Feature Scaling:** Utilized **StandardScaler** to normalize numeric features.
3. **Handling Imbalance:** Implemented `class_weight='balanced'` to handle the minority class effectively.

## 🤖 Models & Evaluation
I implemented and compared multiple machine learning algorithms. Priority was given to **Recall**, as capturing potential churners is critical.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.8230** | **0.5400** | **0.7700** | **0.6300** |
| **SVM** | 0.7855 | 0.4712 | 0.7481 | 0.5782 |
| **KNN** | 0.8100 | 0.5200 | 0.4500 | 0.4800 |

## 📈 Key Insights
Based on the Random Forest model, the top factors driving customer churn are **Age**, **Number of Products**, and **IsActiveMember**.
