# Brain Stroke Classification with Transfer Learning

## Overview

This project applies **Transfer Learning** with several pre-trained CNN architectures to classify brain images into two classes:

- **Stroke**
- **Non-Stroke / Normal**

The notebook experiments with four pre-trained models:

1. DenseNet121
2. ResNet50
3. VGG16
4. MobileNet
5. Xception

The models use **ImageNet pre-trained weights** and replace the original classification head with a binary classification layer.

> **Note:** This README is based directly on the provided notebook and its recorded outputs.

---

## Dataset

The images are organized into training and testing folders:

```text
Brain_Data_Organised/
├── Train/
│   ├── Stroke/
│   └── Normal/
└── Test/
```

### Dataset Statistics

The training directory contains:

| Class | Images |
|---|---:|
| Stroke | 950 |
| Non-Stroke | 1,426 |
| **Total** | **2,376** |

The test set contains **250 images**.

The training set is split into:

- **1,901 images** for training
- **475 images** for validation

The test set contains 250 images.

---

## Data Preparation

Images are loaded using TensorFlow's `image_dataset_from_directory`.

### Image Configuration

```python
image_size = (224, 224)
batch_size = 32
```

The training dataset uses an 80/20 validation split:

```python
validation_split=0.20
```

The test dataset is loaded with:

```python
shuffle=False
```

This allows the true labels to be collected in the same order as the model predictions.

The training and validation datasets are also prefetched:

```python
train_df = train_df.prefetch(buffer_size=32)
val_df = val_df.prefetch(buffer_size=32)
```

---

# Transfer Learning

Transfer Learning is used to take advantage of feature representations learned by CNN models on the **ImageNet** dataset.

Instead of training a deep CNN completely from scratch, the notebook loads pre-trained architectures with:

```python
weights="imagenet"
```

The classification head is then replaced with:

```python
GlobalAveragePooling2D()
Dense(1, activation="sigmoid")
```

The final sigmoid output is suitable for binary classification.

---

# Model Architecture

## 1. DenseNet121

The first model is **DenseNet121**:

```python
base_model = tf.keras.applications.DenseNet121(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet"
)
```

The original classification layer is removed using:

```python
include_top=False
```

Then:

```text
Input (224 × 224 × 3)
        ↓
DenseNet121
        ↓
GlobalAveragePooling2D
        ↓
Dense(1, Sigmoid)
        ↓
Stroke / Non-Stroke
```

The notebook reports **429 layers** in the resulting model.

### Fine-Tuning Configuration

The code sets layers starting from index 351 as trainable:

```python
for layer in base_model.layers[351:]:
    layer.trainable = True
```

The notebook comments describe this as freezing layers from 1 to 350.

---

## 2. ResNet50

The second architecture is **ResNet50**:

```python
base_model = tf.keras.applications.ResNet50(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet"
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

The code sets layers starting from index 100 as trainable:

```python
for layer in base_model.layers[100:]:
    layer.trainable = True
```

---

## 3. VGG16

The third model is **VGG16**:

```python
base_model = tf.keras.applications.VGG16(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet"
)
```

Architecture:

```text
Input (224 × 224 × 3)
        ↓
VGG16
        ↓
GlobalAveragePooling2D
        ↓
Dense(1, Sigmoid)
        ↓
Stroke / Non-Stroke
```

The code sets layers starting from index 100 as trainable.

---

## 4. MobileNet

The fourth architecture is **MobileNet**:

```python
base_model = tf.keras.applications.mobilenet.MobileNet(
    include_top=False,
    weights="imagenet",
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

The code sets layers starting from index 100 as trainable.

---

## 5. Xception

The fifth architecture is **Xception**:

```python
base_model = tf.keras.applications.xception.Xception(
    include_top=False,
    weights="imagenet",
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

The code sets layers starting from index 100 as trainable.

---

# Training Configuration

The common optimizer is Adam:

```python
Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
)
```

Loss function:

```python
binary_crossentropy
```

The models are evaluated using:

- Accuracy
- Precision
- Recall
- Specificity at Sensitivity = 0.5
- Sensitivity at Specificity = 0.5

### Epochs

| Model | Epochs |
|---|---:|
| DenseNet121 | 50 |
| ResNet50 | 50 |
| VGG16 | 50 |
| MobileNet | 50 |
| Xception | 30 |

---

# Callbacks

The notebook defines three callbacks.

### ModelCheckpoint

Saves the best model based on validation loss:

```python
ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_best_only=True
)
```

### ReduceLROnPlateau

Reduces the learning rate when validation loss stops improving:

```python
ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5
)
```

### EarlyStopping

Stops training when the monitored loss does not improve for 10 epochs:

```python
EarlyStopping(
    monitor="loss",
    patience=10
)
```

---

# Results

The models were evaluated on the **250-image test set**.

## DenseNet121

Test results:

| Metric | Value |
|---|---:|
| Loss | 0.0611 |
| Accuracy | **98.40%** |
| Precision | 97.64% |
| Recall | 99.20% |
| Specificity at Sensitivity 0.5 | 100.00% |
| Sensitivity at Specificity 0.5 | 99.20% |

Classification report:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Non-Stroke | 0.9919 | 0.9760 | 0.9839 | 125 |
| Stroke | 0.9764 | 0.9920 | 0.9841 | 125 |
| **Accuracy** | | | **0.9840** | **250** |

---

## ResNet50

Test results:

| Metric | Value |
|---|---:|
| Loss | 0.1050 |
| Accuracy | **96.80%** |
| Precision | 93.98% |
| Recall | 100.00% |
| Specificity at Sensitivity 0.5 | 98.40% |
| Sensitivity at Specificity 0.5 | 100.00% |

Classification report:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Class 0 | 1.0000 | 0.9360 | 0.9669 | 125 |
| Class 1 | 0.9398 | 1.0000 | 0.9690 | 125 |
| **Accuracy** | | | **0.9680** | **250** |

---

## VGG16

The recorded test evaluation produced:

| Metric | Value |
|---|---:|
| Loss | 0.7112 |
| Accuracy | **50.00%** |

Classification report:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Class 0 | 0.5000 | 1.0000 | 0.6667 | 125 |
| Class 1 | 0.0000 | 0.0000 | 0.0000 | 125 |
| **Accuracy** | | | **0.5000** | **250** |

The recorded predictions show that the model predicted only one class on the test set.

---

## MobileNet

Test results:

| Metric | Value |
|---|---:|
| Loss | 0.2731 |
| Accuracy | **96.00%** |
| Precision | 92.59% |
| Recall | 100.00% |
| Specificity at Sensitivity 0.5 | 96.00% |
| Sensitivity at Specificity 0.5 | 100.00% |

Classification report:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Class 0 | 1.0000 | 0.9200 | 0.9583 | 125 |
| Class 1 | 0.9259 | 1.0000 | 0.9615 | 125 |
| **Accuracy** | | | **0.9600** | **250** |

---

## Xception

Test results:

| Metric | Value |
|---|---:|
| Loss | 0.1012 |
| Accuracy | **98.40%** |
| Precision | 96.90% |
| Recall | 100.00% |
| Specificity at Sensitivity 0.5 | 97.60% |
| Sensitivity at Specificity 0.5 | 100.00% |

Classification report:

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Stroke | 1.0000 | 0.9680 | 0.9837 | 125 |
| Non-Stroke | 0.9690 | 1.0000 | 0.9843 | 125 |
| **Accuracy** | | | **0.9840** | **250** |

---

# Model Comparison

Based on the recorded test accuracy:

| Model | Test Accuracy |
|---|---:|
| **DenseNet121** | **98.40%** |
| **Xception** | **98.40%** |
| ResNet50 | 96.80% |
| MobileNet | 96.00% |
| VGG16 | 50.00% |

DenseNet121 and Xception achieved the highest recorded test accuracy at **98.40%**.

---

# Evaluation and Visualization

The notebook generates:

### Training & Validation Accuracy

The training and validation accuracy are plotted across epochs to observe model learning.

### Training & Validation Loss

Training and validation loss are plotted to monitor convergence and possible overfitting.

### Confusion Matrix

A confusion matrix is generated for each model to visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

The confusion matrix is created using:

```python
confusion_matrix(y_true, y_pred)
```

and visualized with Seaborn.

---

# Technologies Used

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

# How to Run

## 1. Open the Notebook

The notebook is designed to run in **Google Colab**.

## 2. Mount Google Drive

The notebook uses:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 3. Prepare the Dataset

Place the dataset in the expected structure:

```text
Brain_Data_Organised/
├── Train/
│   ├── Stroke/
│   └── Normal/
└── Test/
    ├── Stroke/
    └── Normal/
```

## 4. Install Dependencies

The notebook imports TensorFlow/Keras, Scikit-learn, Seaborn, NumPy, Pandas and other dependencies.

It also installs the `utils` package:

```python
!pip install utils
```

## 5. Run the Notebook

Run the cells sequentially to:

1. Load the dataset
2. Create training/validation/test datasets
3. Build the transfer learning models
4. Configure fine-tuning
5. Train the models
6. Save the best checkpoints
7. Evaluate the models
8. Generate classification reports and confusion matrices

---

# Project Workflow

```text
Brain Images
     ↓
Train / Validation / Test Split
     ↓
Resize to 224 × 224
     ↓
Pre-trained CNN with ImageNet Weights
     ↓
Partial Layer Fine-Tuning Configuration
     ↓
Global Average Pooling
     ↓
Dense(1, Sigmoid)
     ↓
Binary Classification
     ↓
Evaluation
     ↓
Accuracy / Precision / Recall / Confusion Matrix
```

---

# Important Note About Layer Freezing

The notebook comments describe the setup as freezing earlier layers and training later layers.

For example:

```python
for layer in base_model.layers[100:]:
    layer.trainable = True
```

However, this code explicitly sets the later layers to trainable; it does **not** explicitly set the earlier layers to `False`.

If the goal is to truly freeze the first 100 layers, the implementation should explicitly use:

```python
for layer in base_model.layers[:100]:
    layer.trainable = False

for layer in base_model.layers[100:]:
    layer.trainable = True
```

The same consideration applies to the DenseNet121 configuration.

This README reports the notebook's actual implementation and recorded results without changing them.

---

# Disclaimer

This project is an educational Deep Learning / Transfer Learning experiment.

The reported accuracy values are based on the specific dataset split and training configuration used in the notebook. They should **not** be interpreted as clinical diagnostic performance or as a replacement for professional medical diagnosis.

---

## Author

**Brain Stroke Classification using Transfer Learning**

Deep Learning / Computer Vision Project
