# Global Sustainable Energy & CO₂ Emissions Analysis

A data analysis and machine learning project that explores global sustainable-energy indicators and builds regression models to **predict CO₂ emissions**.

The project is implemented in `Code.ipynb` using the `global-data-on-sustainable-energy (1).csv` dataset. It covers data exploration, cleaning, feature engineering, scaling, correlation analysis, visualization, feature selection, and regression model comparison.

## Project Overview

The main objective is to analyze the relationship between energy, economic, geographic, and demographic indicators and the target variable:

**`CO2` — CO₂ emissions in kilotons.**

Unlike a classification problem, this project treats CO₂ emissions as a **continuous target** and therefore uses regression algorithms and regression metrics.

## Dataset

The notebook loads:

```text
global-data-on-sustainable-energy (1).csv
```

The initial dataset is organized around the following groups of variables:

- Country/entity and year
- Geographic information
- Electricity access
- Fossil-fuel electricity generation
- Nuclear electricity generation
- Renewable electricity generation
- Renewable-energy indicators
- Energy consumption and intensity
- CO₂ emissions
- GDP growth and GDP per capita

### Main Features

| Feature | Description |
|---|---|
| `Entity` | Country/entity name |
| `Year` | Observation year |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |
| `Land Area(Km2)` | Land area |
| `Density\n(P/Km2)` | Population density |
| `Access to electricity (% of population)` | Percentage of population with electricity access |
| `Access to clean fuels for cooking` | Access to clean cooking fuels |
| `Electricity from fossil fuels (TWh)` | Electricity generated from fossil fuels |
| `Electricity from nuclear (TWh)` | Electricity generated from nuclear sources |
| `Electricity from renewables (TWh)` | Electricity generated from renewable sources |
| `Low-carbon electricity (% electricity)` | Share of electricity from low-carbon sources |
| `Renewable-electricity-generating-capacity-per-capita` | Renewable generation capacity per capita |
| `Renewable energy share in the total final energy consumption (%)` | Renewable share of final energy consumption |
| `Renewables (% equivalent primary energy)` | Renewable contribution to equivalent primary energy |
| `Financial flows to developing countries (US $)` | Financial flows to developing countries |
| `Primary energy consumption per capita (kWh/person)` | Primary energy consumption per person |
| `Energy intensity level of primary energy (MJ/$2017 PPP GDP)` | Primary energy intensity |
| `Value_co2_emissions_kt_by_country` | CO₂ emissions |
| `gdp_growth` | GDP growth |
| `gdp_per_capita` | GDP per capita |

## Data Cleaning

The notebook performs several data-cleaning operations.

### 1. Column Selection and Ordering

The dataset is reordered into a predefined set of columns to make the analysis consistent.

### 2. Missing Values

Missing values are inspected using:

```python
df.isna().sum()
```

A missing-value visualization is also created.

Three columns with a high number of missing values are removed:

```text
Financial flows to developing countries (US $)
Renewables (% equivalent primary energy)
Renewable-electricity-generating-capacity-per-capita
```

### 3. Mean Imputation

Missing values in selected numerical columns are replaced with their column means.

Mean imputation is applied to:

- `Access to clean fuels for cooking`
- `Renewable energy share in the total final energy consumption (%)`
- `Electricity from nuclear (TWh)`
- `Energy intensity level of primary energy (MJ/$2017 PPP GDP)`
- `Value_co2_emissions_kt_by_country`
- `gdp_growth`
- `gdp_per_capita`

Any rows that still contain missing values are then removed with:

```python
df = df.dropna()
```

### 4. Duplicate Rows

The notebook checks for duplicate records using:

```python
df.duplicated().sum()
```

## Feature Engineering

Several columns are renamed to make them easier to use:

```text
Value_co2_emissions_kt_by_country → CO2
Land Area(Km2) → Land
Density\n(P/Km2) → Density
```

The `Density` values are cleaned by removing commas and converting them to integers.

The notebook also creates country/land-area data used for visualization.

## Feature Scaling

`MinMaxScaler` is used to normalize selected numerical features.

Initially, the following columns are scaled:

```text
Electricity from fossil fuels (TWh)
CO2
Land
Electricity from nuclear (TWh)
Electricity from renewables (TWh)
Density
```

Later, after feature selection, the notebook also scales:

```text
Land
Primary energy consumption per capita (kWh/person)
gdp_per_capita
```

The scaled values are stored in `df_scaled`.

## Correlation Analysis

A numerical correlation matrix is calculated using:

```python
correlation_matrix = df.corr(numeric_only=True)
```

A Plotly heatmap is used to visualize relationships between numerical features.

The notebook also identifies:

- The five features most positively correlated with `CO2`
- The five features most negatively correlated with `CO2`

## Feature Selection

The notebook uses a correlation-based feature reduction strategy.

Features with an absolute correlation to `CO2` below `0.5` are considered candidates for removal.

The following columns are protected from being removed by this rule:

```text
gdp_per_capita
Primary energy consumption per capita (kWh/person)
Year
```

The target column `CO2` is also protected.

The resulting reduced dataset is then used for the machine learning stage.

## Data Visualization

The notebook contains multiple visualizations for understanding the data.

### CO₂ Analysis

- Top 15 CO₂-emitting observations
- Top 10 countries by maximum CO₂ emissions
- Maximum CO₂ emissions by year
- CO₂ box plot
- CO₂ relationships with energy/geographic features

### Energy Analysis

- Electricity from fossil fuels by country
- Top 10 fossil-fuel electricity observations
- Fossil-fuel electricity box plot
- Renewable electricity by country
- Renewable electricity box plot
- Country land-area comparisons

### Country-Level CO₂ Trends

CO₂ emissions are visualized over time for:

- Canada
- United States
- China
- Brazil
- Australia

## Encoding

Before model training, the categorical `Entity` column is converted into numerical labels using `LabelEncoder`:

```python
le = LabelEncoder()
df.Entity = le.fit_transform(df.Entity)
```

This allows the country/entity information to be included in the regression models.

## Machine Learning

The target is defined as:

```python
X = df.drop(columns=['CO2'])
y = df['CO2']
```

The data is split into:

- 80% training data
- 20% testing data
- `random_state=42`

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

## Regression Models

The notebook compares three regression algorithms:

### 1. Linear Regression

A baseline linear model used to estimate CO₂ emissions from the selected features.

### 2. Random Forest Regressor

An ensemble tree-based regression model:

```python
RandomForestRegressor(random_state=42)
```

It is also used for feature-importance analysis.

### 3. Gradient Boosting Regressor

A boosting-based regression model:

```python
GradientBoostingRegressor(random_state=42)
```

## Model Evaluation

The models are evaluated using three regression metrics:

### R² Score

Measures how much of the variance in the target is explained by the model.

Higher values are better.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted CO₂ values.

Lower values are better.

### Root Mean Squared Error (RMSE)

Measures prediction error while giving greater weight to larger errors.

Lower values are better.

The notebook calculates:

```python
r2_score(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))
```

## Feature Importance

The notebook uses model feature importance to identify the most influential variables.

The top five features are extracted and displayed in a Plotly bar chart.

This analysis is intended to show which input variables contribute most strongly to the model's CO₂ predictions.

## Forward Selection

The notebook also implements a custom **Forward Selection** procedure.

The function starts with no selected features and iteratively:

1. Tests each remaining feature.
2. Trains the supplied regression model.
3. Calculates the R² score.
4. Selects the feature that gives the strongest improvement.
5. Continues while the improvement exceeds the configured threshold.

The default threshold is:

```python
threshold=0.01
```

This provides another approach to reducing the feature set based on predictive performance rather than correlation alone.

## Post-Selection Modeling

After feature-selection processing, the notebook again compares:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The same evaluation metrics are used:

- R²
- MAE
- RMSE

The model with the highest R² is tracked as the best model.

> **Important:** The notebook code available for this project does not provide a complete final printed result table for all post-selection models, so specific final metric values are not claimed here.

## Project Workflow

```text
Load Dataset
     ↓
Reorder Columns
     ↓
Data Exploration
     ↓
Missing-Value Analysis
     ↓
Remove High-Missing Columns
     ↓
Mean Imputation
     ↓
Remove Remaining Missing Rows
     ↓
Duplicate Check
     ↓
Feature Engineering
     ↓
Scaling
     ↓
Correlation Analysis
     ↓
Correlation-Based Feature Selection
     ↓
Data Visualization
     ↓
Entity Encoding
     ↓
Train/Test Split
     ↓
Regression Models
     ↓
R² / MAE / RMSE Evaluation
     ↓
Feature Importance
     ↓
Forward Selection
     ↓
Post-Selection Model Comparison
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

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyter
```

## How to Run

1. Place the dataset in the same working directory as the notebook:

```text
global-data-on-sustainable-energy (1).csv
```

2. Open the notebook:

```bash
jupyter notebook Code.ipynb
```

3. Run the cells from top to bottom.

You can also open the notebook using JupyterLab or another compatible Jupyter environment.

## Project Structure

```text
.
├── Code.ipynb
├── global-data-on-sustainable-energy (1).csv
└── README.md
```

## Notes

- The target variable is `CO2`, representing CO₂ emissions.
- This is a **regression** project, not a classification project.
- The notebook combines exploratory data analysis with machine learning.
- Missing values are handled through a combination of column removal, mean imputation, and dropping remaining incomplete rows.
- Feature selection is performed using both correlation analysis and forward selection.
- Random Forest is used for feature-importance analysis.
- The notebook contains both scaled and unscaled DataFrame versions; the exact version used in a particular model is determined by the notebook's execution flow.
