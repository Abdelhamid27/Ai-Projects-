# CO2 Emissions Prediction Project 🌍

## 📌 Project Overview
This project aims to analyze and predict **CO2 emissions** across different countries and years. By leveraging historical energy and environmental data, we built a machine learning model capable of identifying the key drivers of carbon emissions and providing accurate forecasts.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Plotly, Matplotlib.
* **Environment:** Jupyter Notebook / VS Code.

---

## 🚀 Key Features & Workflow

### 1. Data Preprocessing & Cleaning
* Handled missing values and standardized feature names.
* Converted categorical data (Countries/Entities) using **Label Encoding**.
* Feature scaling using **MinMaxScaler** to normalize data for better model convergence.

### 2. Exploratory Data Analysis (EDA)
* **Correlation Analysis:** Used Heatmaps to identify the strongest predictors (e.g., Fossil Fuels).
* **Outlier Detection:** Implemented Box Plots to visualize significant emission gaps between nations.
* **Visualization:** Created interactive Scatter Plots to study the relationship between energy sources and CO2 output.

### 3. Feature Selection & Optimization
* Applied **Forward Feature Selection** to reduce model complexity and eliminate noise.
* Used **GridSearchCV** for hyperparameter tuning to find the optimal configuration for the Random Forest model.

---

## 📊 Model Performance
After rigorous testing and optimization, the final model achieved the following results:

| Metric | Score |
| :--- | :--- |
| **R2 Score** | **0.89** |
| **Mean Absolute Error (MAE)** | **17,867.17** |

> **Note:** Our model focuses on minimizing MAE to ensure realistic and logical predictions, successfully avoiding the negative values produced by simpler linear models.

---

## 📂 Project Structure
* `data/`: Contains the dataset used for training.
* `notebooks/`: Includes the step-by-step Jupyter Notebook of the analysis.
* `src/`: Python scripts for preprocessing and modeling.
* `README.md`: Project documentation.

---

## 💡 Conclusion
The study confirms that **Electricity from Fossil Fuels** remains the most significant contributor to CO2 emissions. The optimized **Random Forest Regressor** provides a robust tool for environmental data analysis and policy planning.
