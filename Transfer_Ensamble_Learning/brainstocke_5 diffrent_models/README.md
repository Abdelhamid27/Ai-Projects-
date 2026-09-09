# Brain Stroke Classification using Transfer Learning

## 📌 Overview

This project uses **Transfer Learning** with multiple pre-trained CNN architectures to classify brain images into two classes:

- **Stroke**
- **Non-Stroke / Normal**

The notebook compares five different pre-trained models:

- DenseNet121
- ResNet50
- VGG16
- MobileNet
- Xception

All models use **ImageNet pre-trained weights**, remove the original classification head, and add a binary classification head consisting of `GlobalAveragePooling2D` followed by a single sigmoid output.

> This README is based on the provided notebook and its recorded outputs.

---

## 📂 Dataset

The dataset is organized as follows:

```text
Brain_Data_Organised/
├── Train/
│   ├── Stroke/
│   └── Normal/
└── Test/
```

The notebook reports:

| Dataset/Class | Number of Images |
|---|---:|
| Stroke (Train) | 950 |
| Non-Stroke (Train) | 1,426 |
| **Total Training Images** | **2,376** |
| Validation | 475 |
| Training subset | 1,901 |
| Test set | 250 |

The training directory is split into **80% training** and **20% validation**.

### Image Configuration

```python
image_size = (224, 224)
batch_size = 32
```

The test dataset is loaded with:

```python
shuffle=False
```

This is important because the notebook later compares predictions with the true labels in their original order.

---

# 🔄 Transfer Learning Approach

Transfer Learning allows a CNN that has already learned useful visual features from **ImageNet** to be adapted to the brain-stroke classification task.

The notebook uses:

```python
weights='imagenet'
include_top=False
```

The original classification layers are removed and replaced with:

```text
Pre-trained CNN
      ↓
GlobalAveragePooling2D
      ↓
Dense(1, activation='sigmoid')
      ↓
Binary Classification
```

The notebook sets:

```python
base_model.trainable = True
```

for all five architectures.

Therefore, the pre-trained base models are configured to be trainable rather than explicitly freezing the entire backbone.

---

# 🧠 Models

## 1. DenseNet121

The first model is:

```python
tf.keras.applications.DenseNet121(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
```

The classification head is:

```python
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid', name='Final')(x)
```

The notebook reports:

```text
Number of layers: 429
```

### Training

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Binary Cross-Entropy
- Epochs: 50
- Validation data: 20% of training dataset

### Test Results

| Metric | Result |
|---|---:|
| Loss | 0.0611 |
| Accuracy | **98.40%** |
| Precision | 97.64% |
| Recall | 99.20% |
| Specificity at Sensitivity 0.5 | 100.00% |
| Sensitivity at Specificity 0.5 | 99.20% |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Stroke | 0.9919 | 0.9760 | 0.9839 | 125 |
| Non stroke | 0.9764 | 0.9920 | 0.9841 | 125 |
| **Accuracy** | | | **0.9840** | **250** |

---

# 2. ResNet50

The second model is:

```python
tf.keras.applications.ResNet50(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
```

Architecture:

```text
Input (224 × 224 × 3)
        ↓
ResNet50
        ↓
GlobalAveragePooling2D
        ↓
Dense(1, Sigmoid)
        ↓
Stroke / Non-Stroke
```

### Training

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Binary Cross-Entropy
- Epochs: 50

### Test Results

| Metric | Result |
|---|---:|
| Loss | 0.1050 |
| Accuracy | **96.80%** |
| Precision | 93.98% |
| Recall | 100.00% |
| Specificity at Sensitivity 0.5 | 98.40% |
| Sensitivity at Specificity 0.5 | 100.00% |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| class 0 | 1.0000 | 0.9360 | 0.9669 | 125 |
| class 1 | 0.9398 | 1.0000 | 0.9690 | 125 |
| **Accuracy** | | | **0.9680** | **250** |

The confusion matrix in the notebook uses the labels **Stroke** and **Non stroke**.

---

# 3. VGG16

The third model is:

```python
tf.keras.applications.VGG16(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
```

The classification head is:

```python
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid', name='Final')(x)
```

### Training

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Binary Cross-Entropy
- Epochs: 50

### Test Results

| Metric | Result |
|---|---:|
| Loss | 0.7112 |
| Accuracy | **50.00%** |
| Precision | 0.00% |
| Recall | 0.00% |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| class 0 | 0.5000 | 1.0000 | 0.6667 | 125 |
| class 1 | 0.0000 | 0.0000 | 0.0000 | 125 |
| **Accuracy** | | | **0.5000** | **250** |

The recorded output indicates that the model predicted only one class on the test set.

The notebook also produced `UndefinedMetricWarning` because one class had no predicted samples.

---

# 4. MobileNet

The fourth model is:

```python
tf.keras.applications.mobilenet.MobileNet(
    include_top=False,
    weights='imagenet',
    input_shape=IMG_SHAPE
)
```

Architecture:

```text
Input (224 × 224 × 3)
        ↓
MobileNet
        ↓
GlobalAveragePooling2D
        ↓
Dense(1, Sigmoid)
        ↓
Stroke / Non-Stroke
```

### Training

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Binary Cross-Entropy
- Epochs: 50

### Test Results

| Metric | Result |
|---|---:|
| Loss | 0.2731 |
| Accuracy | **96.00%** |
| Precision | 92.59% |
| Recall | 100.00% |
| Specificity at Sensitivity 0.5 | 96.00% |
| Sensitivity at Specificity 0.5 | 100.00% |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| class 0 | 1.0000 | 0.9200 | 0.9583 | 125 |
| class 1 | 0.9259 | 1.0000 | 0.9615 | 125 |
| **Accuracy** | | | **0.9600** | **250** |

---

# 5. Xception

The fifth model is:

```python
tf.keras.applications.xception.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=IMG_SHAPE
)
```

Architecture:

```text
Input (224 × 224 × 3)
        ↓
Xception
        ↓
GlobalAveragePooling2D
        ↓
Dense(1, Sigmoid)
        ↓
Stroke / Non-Stroke
```

### Training

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Binary Cross-Entropy
- Epochs: 30

### Test Results

| Metric | Result |
|---|---:|
| Loss | 0.1012 |
| Accuracy | **98.40%** |
| Precision | 96.90% |
| Recall | 100.00% |
| Specificity at Sensitivity 0.5 | 97.60% |
| Sensitivity at Specificity 0.5 | 100.00% |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| class 0 | 1.0000 | 0.9680 | 0.9837 | 125 |
| class 1 | 0.9690 | 1.0000 | 0.9843 | 125 |
| **Accuracy** | | | **0.9840** | **250** |

The confusion matrix in the notebook uses the labels **Stroke** and **Non stroke**.

---

# 📊 Model Comparison

The recorded test results are:

| Model | Accuracy | Precision | Recall | Loss |
|---|---:|---:|---:|---:|
| **DenseNet121** | **98.40%** | 97.64% | 99.20% | 0.0611 |
| **Xception** | **98.40%** | 96.90% | 100.00% | 0.1012 |
| ResNet50 | 96.80% | 93.98% | 100.00% | 0.1050 |
| MobileNet | 96.00% | 92.59% | 100.00% | 0.2731 |
| VGG16 | 50.00% | 0.00% | 0.00% | 0.7112 |

### 🏆 Best Recorded Accuracy

Two models achieved the highest recorded test accuracy:

- **DenseNet121 — 98.40%**
- **Xception — 98.40%**

DenseNet121 achieved the lowest recorded test loss among the five models.

---

# ⚙️ Training Configuration

The optimizer used across the models is:

```python
Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
```

The loss function is:

```python
binary_crossentropy
```

The models are evaluated using:

- Accuracy
- Precision
- Recall
- Specificity at Sensitivity = 0.5
- Sensitivity at Specificity = 0.5

---

# 🔁 Callbacks

The notebook defines the following callbacks:

### ModelCheckpoint

Saves the best model according to validation loss:

```python
ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_best_only=True
)
```

### ReduceLROnPlateau

Reduces the learning rate when validation loss stops improving:

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)
```

### EarlyStopping

Stops training when validation loss does not improve for 10 epochs:

```python
EarlyStopping(
    monitor='val_loss',
    patience=10
)
```

---

# 📈 Evaluation & Visualization

For each model, the notebook generates:

### Training and Validation Accuracy

Used to observe how accuracy changes during training.

### Training and Validation Loss

Used to monitor convergence and potential overfitting.

### Confusion Matrix

The notebook uses:

```python
confusion_matrix(y_true, y_pred)
```

and visualizes the result with Seaborn.

### Classification Report

The notebook calculates:

- Precision
- Recall
- F1-score
- Accuracy
- Macro Average
- Weighted Average

---

# 🔬 Prediction Process

Because the final layer uses:

```python
Dense(1, activation='sigmoid')
```

the prediction is converted into a binary class using:

```python
y_pred = model.predict(test_df, verbose=1).round()
```

Therefore, predictions are rounded around the sigmoid threshold of approximately `0.5`.

---

# 🛠️ Technologies & Libraries

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn
- Google Colab

---

# ▶️ How to Run

## 1. Open the Notebook

The notebook is designed to run in **Google Colab**.

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Prepare the Dataset

Place the dataset in:

```text
Brain_Data_Organised/
├── Train/
│   ├── Stroke/
│   └── Normal/
└── Test/
```

## 4. Install the Required Package

The notebook contains:

```python
!pip install utils
```

Other dependencies are imported through TensorFlow/Keras and the Python environment.

## 5. Run the Notebook

Run the cells sequentially to:

1. Mount Google Drive
2. Load dependencies
3. Load the image dataset
4. Create training and validation splits
5. Prepare the test dataset
6. Build the transfer learning models
7. Train each model
8. Save the best checkpoints
9. Load the saved models
10. Evaluate the models
11. Generate classification reports
12. Generate confusion matrices
13. Compare the results

---

# 🔄 Project Workflow

```text
Brain Image Dataset
        ↓
Train / Validation / Test Split
        ↓
Resize Images to 224 × 224
        ↓
Pre-trained CNN + ImageNet Weights
        ↓
Global Average Pooling
        ↓
Dense Layer + Sigmoid
        ↓
Binary Classification
        ↓
Model Evaluation
        ↓
Accuracy / Precision / Recall
        ↓
Classification Report + Confusion Matrix
```

---

# ⚠️ Important Notes

### 1. The notebook uses trainable pre-trained backbones

Each model contains:

```python
base_model.trainable = True
```

So the notebook does not explicitly freeze the complete pre-trained backbone.

### 2. VGG16 performed poorly

VGG16 achieved only **50% test accuracy** in the recorded experiment, while the other models achieved between 96% and 98.4%.

### 3. Results are specific to this experiment

The reported metrics depend on the dataset, split, preprocessing, initialization, training configuration, and hardware/software environment used in the notebook.

### 4. Medical disclaimer

This project is an educational Deep Learning / Computer Vision experiment. The results should **not** be considered clinical diagnostic performance and should not be used as a replacement for professional medical diagnosis.

---

## 👨‍💻 Project

**Brain Stroke Classification with Transfer Learning**

A comparison of five pre-trained CNN architectures for binary brain-image classification.
