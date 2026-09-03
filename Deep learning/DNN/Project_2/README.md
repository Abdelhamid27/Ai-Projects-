# Simple Regression with Artificial Neural Network

## 📌 Project Overview

This project demonstrates how to solve a **regression problem using an
Artificial Neural Network (ANN)** built with **TensorFlow/Keras**.

Instead of using a real-world dataset, the notebook generates a
synthetic regression dataset with Scikit-learn's `make_regression`, then
applies preprocessing, builds a deep fully connected neural network,
trains it, and evaluates its predictions.

The workflow covered in the notebook is:

1.  Generate synthetic regression data
2.  Split the data into training and testing sets
3.  Standardize the input features
4.  Standardize the target values
5.  Build an ANN using multiple Dense layers
6.  Compile the model with Adam and Mean Squared Error
7.  Train the model
8.  Predict values for the test set
9.  Calculate Mean Squared Error
10. Convert predictions back to the original target scale
11. Compare predicted and actual values

## 🎯 Objective

The main objective is to understand how a neural network can learn the
relationship between multiple input features and a **continuous
numerical target**.

This is a regression task, so the output layer contains **one neuron
with no activation function** (linear output).

## 📊 Dataset

The notebook creates the dataset using:

``` python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1
)
```

### Dataset characteristics

-   **Samples:** 1,000
-   **Input features:** 10
-   **Target:** Continuous numerical value
-   **Noise:** 0.1
-   **Training samples:** 800
-   **Testing samples:** 200

The data is split using:

``` python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## 🛠️ Technologies & Libraries

-   Python
-   NumPy
-   TensorFlow
-   Keras
-   Scikit-learn

### Main imports

``` python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
```

## 🔄 Data Preprocessing

### 1. Train/Test Split

The dataset is divided into:

-   80% training data
-   20% testing data

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 2. Feature Scaling

The input features are standardized using `StandardScaler`:

``` python
scaler_X = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
```

Notice that `fit_transform()` is applied only to the training data,
while the test data uses `transform()`. This avoids fitting the scaler
using information from the test set.

### 3. Target Scaling

The target variable is also standardized:

``` python
scaler_y = StandardScaler()

y_train = scaler_y.fit_transform(
    y_train.reshape(-1, 1)
).flatten()

y_test = scaler_y.transform(
    y_test.reshape(-1, 1)
).flatten()
```

The target is reshaped to a 2D array because `StandardScaler` expects 2D
input, then flattened back to a one-dimensional array for model
training.

## 🧠 ANN Architecture

The notebook builds a fully connected neural network using Keras
`Sequential`.

Architecture:

``` text
Input: 10 features
        ↓
Dense: 64 neurons + ReLU
        ↓
Dense: 32 neurons + ReLU
        ↓
Dense: 16 neurons + ReLU
        ↓
Dense: 16 neurons + ReLU
        ↓
Dense: 16 neurons + ReLU
        ↓
Dense: 8 neurons + ReLU
        ↓
Dense: 8 neurons + ReLU
        ↓
Output: 1 neuron + Linear
```

### Model Code

``` python
model = Sequential()

model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
```

### Why is there no activation function in the output layer?

This is a **regression** problem. The model needs to predict a
continuous numerical value, so the final neuron uses the default
**linear activation**.

A linear output allows the model to produce unrestricted continuous
values.

## ⚙️ Model Compilation

The model is compiled with:

``` python
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)
```

### Configuration

-   **Optimizer:** Adam
-   **Loss Function:** Mean Squared Error (MSE)

MSE measures the average squared difference between the predicted and
actual target values.

## 🏋️ Model Training

The notebook trains the ANN using:

``` python
model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=32,
    verbose=1,
    validation_split=0.2
)
```

### Training configuration

  Parameter                         Value
  ------------------ --------------------
  Epochs                              300
  Batch Size                           32
  Validation Split                    20%
  Optimizer                          Adam
  Loss                 Mean Squared Error

The `validation_split=0.2` means that 20% of the training data is used
for validation during training.

## 🧪 Prediction & Evaluation

After training, the model predicts the test data:

``` python
y_pred = model.predict(X_test)
```

The notebook checks the shapes of the predictions and test targets
before calculating the evaluation metric.

### Mean Squared Error

``` python
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")
```

Because both `y_test` and `y_pred` are currently in the standardized
target scale, this MSE is calculated on the **scaled target values**.

## 🔄 Returning Predictions to Original Scale

The notebook uses the fitted target scaler to convert predictions and
actual values back to their original scale:

``` python
y_pred_original = scaler_y.inverse_transform(
    y_pred
).flatten()

y_test_original = scaler_y.inverse_transform(
    y_test.reshape(-1, 1)
).flatten()
```

This makes the predictions easier to interpret because they are returned
to the same scale as the original target variable.

The notebook then displays the first five predicted and actual values:

``` text
Predicted vs Actual
```

## 📁 Project Structure

``` text
.
├── Simple_regression.ipynb
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

``` bash
pip install numpy tensorflow scikit-learn
```

### 3. Open the notebook

You can run the notebook using:

-   Jupyter Notebook
-   JupyterLab
-   Google Colab
-   VS Code with the Jupyter extension

### 4. Run the cells in order

Run the cells sequentially from dataset generation through evaluation.

## 📚 Key Deep Learning Concepts

This notebook demonstrates:

-   Artificial Neural Networks (ANN)
-   Dense / Fully Connected Layers
-   ReLU activation
-   Linear output layer
-   Regression with Neural Networks
-   Feature Standardization
-   Target Standardization
-   Train/Test Split
-   Validation Split
-   Adam optimizer
-   Mean Squared Error
-   Epochs
-   Batch Size
-   Model Prediction
-   Inverse Scaling

## 💡 Important Learning Point

One of the most important concepts demonstrated here is the difference
between **classification and regression** in the output layer.

For this regression problem:

``` python
Dense(1)
```

is used instead of something like:

``` python
Dense(10, activation='softmax')
```

The model produces a single continuous numerical prediction rather than
a class probability.

## 👨‍💻 Author

**Abdelhamid Ibrahim Abdelhamid**

Computer Science & Artificial Intelligence Graduate\
Interested in Data Science, Machine Learning & Deep Learning
