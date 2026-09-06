# Breast Cancer Classification using Artificial Neural Network (ANN) 🧠

A Deep Learning project that uses an **Artificial Neural Network (ANN)** with **Keras** to classify breast cancer cases as **Malignant (M)** or **Benign (B)** using the Breast Cancer Wisconsin (Diagnostic) dataset.

The notebook is designed as an introduction to ANN concepts and covers the complete workflow from data preparation to model evaluation.

---

## 📌 Project Overview

This project demonstrates how an Artificial Neural Network can be applied to a binary classification problem using structured/tabular medical data.

The workflow includes:

- Loading and exploring the dataset
- Selecting features and target
- Encoding categorical labels
- Splitting the data into training and testing sets
- Standardizing numerical features
- Building an ANN using Keras
- Applying Dropout for regularization
- Training the model
- Visualizing training and validation performance
- Generating predictions
- Evaluating the model with a confusion matrix and classification report

---

## 🎯 Objective

The objective is to predict whether a breast mass is:

- **Malignant (M)**
- **Benign (B)**

The diagnosis column is encoded numerically using `LabelEncoder` before training the neural network.

---

## 📊 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) Data Set**.

The features are computed from digitized images of fine needle aspirate (FNA) samples of breast masses and describe characteristics of cell nuclei.

### Dataset Information

- **569 samples**
- **32 columns**
  - 1 ID column
  - 1 diagnosis column
  - 30 numerical features
- **357 benign cases**
- **212 malignant cases**
- No missing attribute values are reported in the dataset description.

### Features

The dataset contains measurements related to:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

For each characteristic, three types of measurements are provided:

- Mean
- Standard Error (`SE`)
- Worst

This results in **30 input features**.

---

## 🔄 Data Preprocessing

### 1. Feature and Target Selection

The ID column is excluded from the model input.

```python
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values
```

This gives the ANN **30 input features**.

### 2. Label Encoding

The diagnosis labels are converted from categorical values into numerical values using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)
```

### 3. Train-Test Split

The data is split using:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=0
)
```

The resulting training set contains:

```text
512 samples × 30 features
```

The remaining **57 samples** are used as the test set.

### 4. Feature Scaling

`StandardScaler` is used to standardize the numerical features.

```python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

The scaler is fitted on the training data and then applied to the test data.

---

## 🧠 ANN Architecture

The model is built using Keras `Sequential`.

### Network Architecture

```text
Input
  │
  ▼
Dense Layer — 16 neurons
ReLU activation
  │
  ▼
Dropout — 10%
  │
  ▼
Dense Layer — 16 neurons
ReLU activation
  │
  ▼
Dropout — 10%
  │
  ▼
Dense Layer — 16 neurons
ReLU activation
  │
  ▼
Dropout — 10%
  │
  ▼
Output Layer — 1 neuron
Sigmoid activation
```

### Architecture Details

| Layer | Units | Activation | Dropout |
|---|---:|---|---:|
| Dense 1 | 16 | ReLU | 10% |
| Dense 2 | 16 | ReLU | 10% |
| Dense 3 | 16 | ReLU | 10% |
| Output | 1 | Sigmoid | — |

The model contains **1,057 trainable parameters**.

### Why ReLU?

ReLU is used in the hidden layers to introduce non-linearity and allow the network to learn complex relationships between the input features and target.

### Why Sigmoid?

The output layer contains one neuron with a sigmoid activation function because this is a **binary classification** problem. The model produces a value between 0 and 1, and the notebook uses a threshold of `0.5` for the final class prediction.

---

## 🛡️ Dropout

A dropout rate of **0.1 (10%)** is applied after every hidden layer.

```python
Dropout(rate=0.1)
```

Dropout is included in the model to help prevent overfitting during training.

---

## ⚙️ Model Compilation

The ANN is compiled using:

```python
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Optimizer

**Adam** is used for gradient-based optimization.

### Loss Function

**Binary Cross-Entropy** is used because the task has two possible classes.

### Evaluation Metric

**Accuracy** is used during training to monitor classification performance.

---

## 🏋️ Model Training

The ANN is trained with the following configuration:

| Parameter | Value |
|---|---:|
| Epochs | 200 |
| Batch Size | 32 |
| Validation Split | 20% |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |

```python
history = classifier.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=200
)
```

The notebook states that the batch size and number of epochs were selected using **trial and error**.

---

## 📈 Training Performance

The training history is used to visualize:

### Accuracy

The project plots:

- Training Accuracy
- Validation Accuracy

### Loss

The project also plots:

- Training Loss
- Validation Loss

At the final training epoch:

```text
Training Accuracy:   100.00%
Training Loss:       0.0045
Validation Accuracy: 99.03%
Validation Loss:     0.0090
```

---

## 🧪 Model Evaluation

After training, predictions are generated for the held-out test set.

```python
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)
```

The project evaluates the model using:

- Confusion Matrix
- Precision
- Recall
- F1-Score
- Accuracy

---

## 📊 Results

The ANN achieved the following results on the **57-sample test set**:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 0 | 1.0000 | 0.9714 | 0.9855 | 35 |
| 1 | 0.9565 | 1.0000 | 0.9778 | 22 |
| **Accuracy** | | | **0.9825** | **57** |

### Overall Metrics

- **Test Accuracy:** 98.25%
- **Macro F1-Score:** 98.16%
- **Weighted F1-Score:** 98.25%

A confusion matrix is also generated in the notebook to visualize correct and incorrect classifications.

---

## 🛠️ Technologies & Libraries

- **Python**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Seaborn** — confusion matrix visualization
- **Scikit-learn** — preprocessing, train-test split, and evaluation metrics
- **Keras** — building and training the ANN

---

## 📁 Project Structure

```text
Breast-Cancer-ANN/
│
├── data.csv
├── intro_to_keras_with_breast_cancer_data_ann.ipynb
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras
```

### 3. Make sure the dataset is available

Place:

```text
data.csv
```

in the same directory as the notebook.

### 4. Open the notebook

```bash
jupyter notebook intro_to_keras_with_breast_cancer_data_ann.ipynb
```

Run the notebook cells in order.

---

## 🔍 Evaluation Metrics

### Accuracy

Measures the percentage of predictions that are correctly classified.

### Precision

Measures how many of the samples predicted as a class actually belong to that class.

### Recall

Measures how many of the actual samples of a class are correctly identified.

### F1-Score

Combines precision and recall into a single metric.

### Confusion Matrix

Provides a visual summary of correct and incorrect predictions for each class.

---

## 💡 Key Takeaways

- ANN can be applied to structured/tabular classification problems.
- Feature scaling is an important preprocessing step for neural networks.
- ReLU is used in the hidden layers to learn non-linear relationships.
- Sigmoid is used for the binary classification output.
- Dropout is used as a regularization technique.
- The recorded model achieved **98.25% test accuracy** on the held-out test set.

---

## ⚠️ Disclaimer

This is an educational Deep Learning project demonstrating ANN-based classification. The reported performance comes from the specific train/test split and training run recorded in the notebook and should **not** be interpreted as clinical diagnostic performance.

---

## 📚 Dataset Reference

The notebook identifies the dataset as the **Breast Cancer Wisconsin (Diagnostic) Data Set** and references the UCI Machine Learning Repository as a source for the dataset information.
