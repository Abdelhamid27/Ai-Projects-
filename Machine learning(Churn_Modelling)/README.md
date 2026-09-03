# Customer Churn Prediction

A machine learning project for predicting whether a bank customer will **churn (exit)** based on demographic, financial, and account-related features.

The project is implemented in the `Churn_Modelling.ipynb` Jupyter Notebook and covers data exploration, preprocessing, feature encoding, scaling, model training, evaluation, confusion matrices, and ROC-AUC comparison.

## Project Overview

Customer churn prediction is a binary classification problem:

- `0` → Customer retained
- `1` → Customer churned / exited

The notebook uses the `Churn_Modelling.csv` dataset and evaluates several classification algorithms to identify a suitable model for predicting customer churn.

## Dataset

The dataset contains **10,000 customer records** and 14 original columns.

### Original Features

| Feature | Description |
|---|---|
| `RowNumber` | Row identifier |
| `CustomerId` | Unique customer identifier |
| `Surname` | Customer surname |
| `CreditScore` | Customer credit score |
| `Geography` | Customer country |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Tenure` | Number of years with the bank |
| `Balance` | Customer account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated customer salary |
| `Exited` | Target variable indicating churn |

The notebook reports no missing values and no duplicate rows. The target contains:

- `0`: 7,963 customers
- `1`: 2,037 customers

This indicates that the target is imbalanced, with churn representing the minority class.

## Data Preprocessing

The notebook performs the following preprocessing steps.

### 1. Remove Identifier Columns

The following columns are removed because they are identifiers/name information rather than useful predictive features:

```python
drop_cols = ['RowNumber', 'CustomerId', 'Surname']
df.drop(drop_cols, axis=1, inplace=True)
```

### 2. Encode Categorical Features

#### Gender

`Gender` is converted to numerical values using `LabelEncoder`.

#### Geography

`Geography` is transformed using `OneHotEncoder` with `drop='first'`, producing:

- `Geography_Germany`
- `Geography_Spain`

The original `Geography` column is then removed.

### 3. Train/Test Split

The processed data is split into training and testing sets:

- Training set: 8,000 samples
- Test set: 2,000 samples
- Test size: 20%
- `random_state=42`

### 4. Feature Scaling

Numerical features are standardized using `StandardScaler`.

The scaler is fitted on the training data and then applied to both the training and test sets.

## Exploratory Data Analysis

The notebook includes several EDA visualizations, including:

- Churn distribution
- Feature distributions
- Relationship between `Balance` and churn
- Relationship between `NumOfProducts` and churn
- Gender distribution by churn status
- Correlation heatmap

These visualizations are used to inspect feature distributions and relationships with the target variable.

## Machine Learning Models

Five classification models are trained and evaluated:

1. **K-Nearest Neighbors (KNN)**
2. **Decision Tree**
3. **Random Forest**
4. **Support Vector Machine (SVM)**
5. **Gaussian Naive Bayes**

The models are trained on the same train/test split to allow a direct comparison.

## Model Evaluation

The notebook evaluates the models using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

Because churn is the minority class, **Recall, F1-Score, and ROC-AUC** are particularly useful alongside accuracy.

## Results

The evaluation performed in the notebook produced the following results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| K-Nearest Neighbors | 0.8300 | 0.6109 | 0.3715 | 0.4620 | 0.7604 |
| Decision Tree | 0.7840 | 0.4550 | 0.5013 | 0.4770 | 0.6772 |
| Random Forest | **0.8665** | 0.7647 | **0.4631** | **0.5769** | **0.8572** |
| SVM | 0.8560 | **0.7692** | 0.3817 | 0.5102 | 0.8248 |
| Naive Bayes | 0.8335 | 0.6351 | 0.3588 | 0.4585 | 0.8044 |

Based on the notebook's reported metrics, **Random Forest** provides the strongest overall performance, achieving the highest accuracy, recall, F1-score, and ROC-AUC among the evaluated models.

The notebook also includes a Random Forest configuration using:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
```

For that evaluation, the reported accuracy was `0.866`, with precision `0.766`, recall `0.458`, and F1-score `0.573`.

## Project Workflow

```text
Load Dataset
     ↓
Data Inspection
     ↓
EDA & Visualization
     ↓
Remove Identifier Columns
     ↓
Encode Categorical Features
     ↓
Train/Test Split
     ↓
Standard Scaling
     ↓
Train Multiple ML Models
     ↓
Evaluate Models
     ↓
Compare Metrics
     ↓
ROC-AUC & Confusion Matrix Analysis
```

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

## Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
```

## How to Run

1. Make sure `Churn_Modelling.csv` is available in the same working directory as the notebook.
2. Open the notebook:

```bash
jupyter notebook Churn_Modelling.ipynb
```

3. Run the notebook cells from top to bottom.

Alternatively, open `Churn_Modelling.ipynb` using JupyterLab or another compatible notebook environment.

## Project Structure

```text
.
├── Churn_Modelling.ipynb
├── Churn_Modelling.csv
└── README.md
```

## Notes

- The notebook focuses on binary customer churn classification.
- The original dataset contains 14 columns; after removing three identifier columns and encoding `Geography`, the modeling dataset contains 11 input features plus the target.
- The target is imbalanced, so accuracy alone should not be used to judge model quality.
- The notebook compares multiple classical machine learning algorithms rather than relying on a single model.
