# 🚢 Ship Type Classification using MobileNetV2

A Deep Learning computer vision project for **classifying ship images into five different ship categories**.

The project uses **Transfer Learning with MobileNetV2**, followed by a dedicated classification head and a fine-tuning stage. The trained model is also intended to be used through a **Streamlit interface** for image-based prediction.

---

## 🎯 Project Objective

The goal of this project is to build an image classification model that can identify the type of ship from an input image.

The model predicts one of five classes:

| Category | Ship Type |
|---:|---|
| `1` | Cargo |
| `2` | Military |
| `3` | Carrier |
| `4` | Cruise |
| `5` | Tankers |

---

## 📊 Dataset

The project uses a ship image dataset containing image filenames and their corresponding categories.

The training metadata is stored in:

```text
train.csv
```

and the test metadata in:

```text
test_ApKoW4T.csv
```

The training CSV contains:

```text
image
category
```

while the test CSV contains:

```text
image
```

### Training Class Distribution

The recorded training distribution is:

| Ship Type | Samples |
|---|---:|
| Cargo | 2,120 |
| Tankers | 1,217 |
| Military | 1,167 |
| Carrier | 916 |
| Cruise | 832 |
| **Total** | **6,252** |

The dataset is imbalanced, with Cargo being the largest class and Cruise being the smallest.

To address this during training, **class weights** were calculated using `compute_class_weight` and passed to `model.fit()`.

---

# 🔄 Project Workflow

```text
Ship Images
     ↓
Load train.csv / test CSV
     ↓
Explore class distribution
     ↓
Check missing values
     ↓
Extract / organize image files
     ↓
ImageDataGenerator
     ↓
Resize to 224 × 224
     ↓
Normalize pixel values
     ↓
Data Augmentation
     ↓
Train / Validation Split
     ↓
Class Weight Calculation
     ↓
MobileNetV2 Transfer Learning
     ↓
Classification Head
     ↓
Phase 1 Training
     ↓
Phase 2 Fine-Tuning
     ↓
Evaluation
     ↓
Prediction / Submission
     ↓
Streamlit Application
```

---

# 🧹 Data Preparation

The notebook performs several preparation steps before training.

### Missing Values

The training and test metadata were checked for missing values.

Recorded result:

```text
Missing values in training data: 0
Missing values in test data:     0
```

### Image Size

All images are resized to:

```text
224 × 224
```

with 3 RGB channels.

### Batch Size

```text
32
```

---

# 🖼️ Data Augmentation

Data augmentation is applied to the training generator to provide additional variation during training.

The validation and test data are not augmented.

The image generator uses pixel rescaling:

```python
rescale=1./255
```

and the training generator applies image augmentation.

This helps the model learn more robust visual features from the ship images.

---

# ⚖️ Class Weights

Because the dataset is imbalanced, class weights are calculated using:

```python
compute_class_weight(
    'balanced',
    classes=classes,
    y=train_generator.classes
)
```

Recorded class weights:

| Class | Weight |
|---:|---:|
| Cargo | 0.5906 |
| Military | 1.0620 |
| Carrier | 1.3780 |
| Cruise | 1.4909 |
| Tankers | 1.0324 |

These weights are passed to the training process so that underrepresented classes receive greater importance during optimization.

---

# 🧠 Model Architecture

The project uses **MobileNetV2 pretrained on ImageNet** as the feature extractor.

```python
MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
```

The original classification head is removed and replaced with a custom head.

### Architecture

```text
Input Image
224 × 224 × 3
      ↓
Rescaling
[0,1] → [-1,1]
      ↓
MobileNetV2
ImageNet Weights
      ↓
Global Average Pooling
      ↓
Batch Normalization
      ↓
Dense Layer
128 neurons + ReLU
      ↓
Dropout
50%
      ↓
Dense Output Layer
5 neurons + Softmax
      ↓
Ship Type
```

### Model Parameters

Recorded model summary:

| Parameter | Value |
|---|---:|
| Total Parameters | 2,427,717 |
| Trainable Parameters – Phase 1 | 167,173 |
| Non-trainable Parameters – Phase 1 | 2,260,544 |

---

# 🔒 Phase 1 – Transfer Learning

In the first training phase, the MobileNetV2 pretrained base is frozen:

```python
base.trainable = False
```

Only the newly added classification layers are trained.

### Configuration

```text
Optimizer: Adam
Learning Rate: 0.001
Loss: Categorical Cross-Entropy
Epochs: 15
Batch Size: 32
```

Class weights are used during training.

### Phase 1 Result

The recorded best validation performance at the end of Phase 1 was:

```text
Validation Accuracy: 87.76%
Validation Loss:     0.3498
```

The best model was saved as:

```text
best_ship_model.keras
```

---

# 🔧 Phase 2 – Fine-Tuning

After the initial transfer-learning stage, fine-tuning was performed.

The MobileNetV2 base was unfrozen and only the **last 30 layers** were made trainable:

```python
base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False
```

A much smaller learning rate was used:

```text
Learning Rate: 0.00001
```

The model was trained for up to 15 epochs with the same validation data and class weights.

### Early Stopping

The fine-tuning phase stopped early after 5 epochs because the validation loss did not improve.

The callback restored the best weights from the fine-tuning phase.

---

# ⚙️ Training Callbacks

Three callbacks were used:

### ModelCheckpoint

```python
ModelCheckpoint(
    'best_ship_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)
```

### EarlyStopping

```text
Monitor: val_loss
Patience: 5
Restore best weights: True
```

### ReduceLROnPlateau

```text
Monitor: val_loss
Factor: 0.2
Patience: 2
Minimum Learning Rate: 1e-6
```

These callbacks help control the learning process and reduce unnecessary training when validation performance stops improving.

---

# 📈 Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

The validation set contains:

```text
1,250 images
```

---

# 📊 Validation Results

The recorded classification report after the notebook's fine-tuning stage was:

| Ship Type | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Cargo | 0.93 | 0.61 | 0.74 | 426 |
| Military | 0.99 | 0.83 | 0.91 | 225 |
| Carrier | 0.99 | 0.90 | 0.94 | 190 |
| Cruise | 0.95 | 0.93 | 0.94 | 161 |
| Tankers | 0.52 | 0.94 | 0.67 | 248 |
| **Overall Accuracy** | | | **0.80** | **1,250** |

### Overall Metrics

```text
Accuracy: 80%
Macro Precision: 88%
Macro Recall:    84%
Macro F1:        84%
Weighted Precision: 87%
Weighted Recall:    80%
Weighted F1:        81%
```

A confusion matrix is also generated to visualize the classification performance across the five ship categories.

---

# 🧪 Test Set Prediction

The notebook also performs predictions on the separate test dataset.

The test generator uses:

```python
ImageDataGenerator(rescale=1./255)
```

without augmentation.

The test images are loaded with:

```text
Target Size: 224 × 224
Batch Size: 32
Shuffle: False
```

The recorded test set contains:

```text
2,680 images
```

Predictions are converted back to the original category numbers:

```text
1 → Cargo
2 → Military
3 → Carrier
4 → Cruise
5 → Tankers
```

A submission file is generated:

```text
submission.csv
```

with the structure:

```text
image,category
```

Example:

```text
1007700.jpg,4
1011369.jpg,4
1051155.jpg,4
1062001.jpg,2
1069397.jpg,4
```

---

# 🌐 Streamlit Application

A **Streamlit application** was added to provide an interactive interface for the trained ship classification model.

The application is intended to make the model easier to use without running the complete training notebook.

### Application Workflow

```text
Upload Ship Image
       ↓
Image Preprocessing
       ↓
Trained MobileNetV2 Model
       ↓
Prediction
       ↓
Predicted Ship Type
```

The Streamlit interface can be used to demonstrate the trained computer vision model through an interactive web application.

> The exact Streamlit implementation depends on the application files included with the project repository.

---

# 🛠️ Technologies Used

### Programming

- Python

### Deep Learning

- TensorFlow
- Keras
- MobileNetV2
- Transfer Learning
- Fine-Tuning

### Data Processing

- Pandas
- NumPy
- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

### Deployment / Interface

- Streamlit

### Environment

- Google Colab
- Google Drive

---

# 📁 Project Structure

A suggested repository structure is:

```text
Ship-Classification/
│
├── train/
│   ├── train.csv
│   └── images/
│
├── test_ApKoW4T.csv
│
├── Ships.ipynb
│
├── best_ship_model.keras
│
├── submission.csv
│
├── app.py
│
└── README.md
```

---

# 🚀 How to Run the Notebook

### 1. Open the Notebook

Open:

```text
Ships.ipynb
```

in Google Colab.

### 2. Mount Google Drive

The notebook uses Google Drive to access the dataset.

### 3. Prepare the Dataset

The notebook expects the training CSV and image archive in the configured Google Drive paths.

### 4. Run the Notebook

Run the cells sequentially to:

1. Load the metadata
2. Explore the dataset
3. Check missing values
4. Extract the images
5. Create image generators
6. Calculate class weights
7. Build MobileNetV2
8. Train the classification head
9. Fine-tune the last 30 layers
10. Evaluate the model
11. Generate predictions
12. Create `submission.csv`

---

# 🚀 Run the Streamlit App

After preparing the trained model and Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser and provide an interface for making ship-type predictions.

---

# 📌 Key Takeaways

- This project solves a **5-class ship image classification** problem.
- **MobileNetV2 with ImageNet weights** is used for transfer learning.
- The dataset contains **6,252 labeled training images** and **2,680 test images**.
- The training data is imbalanced, so **class weights** are used.
- Images are resized to **224 × 224** and normalized.
- The model uses **Global Average Pooling, Batch Normalization, Dense(128), Dropout(0.5), and Softmax**.
- The project uses a two-stage training strategy:
  - Transfer learning with the MobileNetV2 base frozen.
  - Fine-tuning of the last 30 MobileNetV2 layers.
- The best recorded Phase 1 validation accuracy was **87.76%**.
- The recorded classification report after the fine-tuning stage shows **80% validation accuracy**.
- The model generates predictions for the test images and creates `submission.csv`.
- A **Streamlit interface** was added to make the trained model accessible through an interactive application.

---

## ⚠️ Note

The metrics in this README are taken from the recorded outputs of the provided notebook.

The reported **87.76%** refers to the best validation accuracy reached during Phase 1, while the **80%** classification report is the evaluation recorded after the subsequent fine-tuning stage. These are therefore reported separately rather than presented as the same metric.

---

## 👨‍💻 Project

**Ship Type Classification using Deep Learning & Transfer Learning**

Built with Python, TensorFlow/Keras, MobileNetV2, Scikit-learn, and Streamlit.
