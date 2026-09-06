# Introduction to CNN with Keras -- MNIST Digit Classification

## 📌 Overview

This notebook implements a **Convolutional Neural Network (CNN)** using
**Keras/TensorFlow** to recognize handwritten digits from the **MNIST
dataset**.

The project covers the complete deep learning workflow:

-   Loading and exploring image data
-   Checking for missing values
-   Normalizing pixel values
-   Reshaping images for CNN input
-   One-hot encoding labels
-   Splitting data into training, validation, and test sets
-   Building a CNN for image classification
-   Using Dropout for regularization
-   Applying Data Augmentation
-   Using learning-rate reduction and Early Stopping
-   Saving the best model
-   Evaluating the model using accuracy, loss curves, confusion matrix,
    and classification report
-   Inspecting misclassified images
-   Generating predictions and a Kaggle-style submission file

## 🎯 Objective

The objective is to classify handwritten digit images into one of **10
classes (0--9)** using a CNN.

## 📊 Dataset

The notebook expects two CSV files:

-   `train.csv`
-   `test.csv`

The images are stored as flattened pixel vectors. Each image contains
**28 × 28 = 784 pixels**.

The images are grayscale, so the CNN input shape is:

``` text
(28, 28, 1)
```

The training labels are stored in the `label` column.

## 🔄 Data Preparation

### 1. Load the data

``` python
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

The label column is separated from the training features.

### 2. Check missing values

The notebook checks both training and test data for missing values and
reports that no missing values are present.

### 3. Normalization

Pixel values originally range from **0 to 255**. They are divided by 255
to scale them to:

``` text
0 → 1
```

### 4. Reshape images

The flattened 784-pixel vectors are reshaped into:

``` text
28 × 28 × 1
```

The last dimension represents the grayscale channel.

### 5. One-Hot Encoding

The digit labels are converted into 10-dimensional one-hot vectors.

For example:

``` text
2 → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```

### 6. Train / Validation / Test Split

The notebook first creates a validation set from the training data and
then creates an additional test split from the remaining training
portion.

`random_state=2` is used for reproducibility.

## 🧠 CNN Architecture

The CNN uses the following structure:

``` text
Input: 28 × 28 × 1

Conv2D: 32 filters, 5×5, ReLU
Conv2D: 32 filters, 5×5, ReLU
MaxPooling2D: 2×2
Dropout: 25%

Conv2D: 64 filters, 3×3, ReLU
Conv2D: 64 filters, 3×3, ReLU
MaxPooling2D: 2×2
Dropout: 25%

Flatten

Dense: 256, ReLU
Dropout: 50%

Output: Dense 10, Softmax
```

### 🔹 Convolutional Layers

`Conv2D` layers learn visual features from the images using learnable
filters.

Early convolutional layers can learn simpler patterns, while deeper
layers can combine them into more useful representations.

### 🔹 Max Pooling

`MaxPool2D` reduces the spatial dimensions of the feature maps, helping
reduce computational cost and retain strong activations.

### 🔹 Dropout

Dropout is used as a regularization technique to reduce overfitting by
randomly disabling a proportion of neurons during training.

### 🔹 Flatten

`Flatten` converts the final feature maps into a one-dimensional vector
before passing them to the fully connected layers.

### 🔹 Softmax Output

The final Dense layer contains **10 neurons**, one for each digit from 0
to 9.

`softmax` produces a probability distribution over the ten classes.

## ⚙️ Model Compilation

The notebook compiles the model using:

``` text
Optimizer: Adam
Loss: categorical_crossentropy
Metric: accuracy
```

An RMSprop optimizer is also instantiated in the notebook, but the
actual `model.compile()` call uses **Adam**.

## 🛠️ Learning Rate Scheduling

The notebook uses `ReduceLROnPlateau`.

The learning rate is reduced by a factor of **0.5** when the validation
loss stops improving for 5 epochs.

A minimum learning rate of:

``` text
0.00001
```

is used.

## ⏹️ Early Stopping

`EarlyStopping` monitors validation loss.

The training stops when validation loss does not improve for **10
epochs**, and the best weights are restored.

## 💾 Model Checkpoint

`ModelCheckpoint` saves the best-performing model as:

``` text
CNN.keras
```

Only the model with the best validation loss is saved.

## 🧪 Data Augmentation

The notebook uses `ImageDataGenerator` to create variations of training
images.

The applied transformations include:

-   Rotation up to 10 degrees
-   Zoom up to 10%
-   Horizontal shift up to 10%
-   Vertical shift up to 10%

Horizontal and vertical flips are disabled because flipping handwritten
digits could change their meaning.

For example, flipping digits can create ambiguity between digits such as
**6 and 9**.

## 📈 Training

The configured training parameters are:

``` text
Epochs: 100
Batch size: 64
Validation data: X_val, Y_val
```

Training is performed using augmented batches:

``` python
datagen.flow(X_train, Y_train, batch_size=batch_size)
```

## 📊 Evaluation

The notebook evaluates the model using:

### Training and Validation Curves

It plots:

-   Training loss
-   Validation loss
-   Training accuracy
-   Validation accuracy

These curves help identify learning behavior and possible overfitting.

### Confusion Matrix

A confusion matrix is generated to compare:

``` text
True labels vs Predicted labels
```

This makes it easier to identify which digits are frequently confused.

### Classification Report

The notebook also generates a classification report containing standard
classification metrics for each digit class.

## 🔍 Error Analysis

The notebook identifies incorrectly classified test images:

``` python
errors = (Y_pred_classes != Y_true)
```

It then displays several misclassified images along with:

-   Predicted label
-   True label

This provides a simple visual error-analysis step.

## 🚀 Prediction & Submission

After loading the saved `CNN.keras` model, predictions are generated for
the separate test dataset.

The class with the highest probability is selected using:

``` python
np.argmax(results, axis=1)
```

The results are saved to:

``` text
cnn_mnist_datagen.csv
```

with the columns:

``` text
ImageId
Label
```

## 📁 Expected Project Structure

``` text
project/
│
├── introduction-to-cnn-keras using Mnist dataset.ipynb
├── train.csv
├── test.csv
├── CNN.keras
├── cnn_mnist_datagen.csv
└── README.md
```

`CNN.keras` and `cnn_mnist_datagen.csv` are generated by the notebook
after the relevant cells are executed.

## 📦 Main Technologies

-   Python
-   TensorFlow / Keras
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   Scikit-learn

## 🧩 Main Deep Learning Concepts

This project demonstrates:

-   Convolutional Neural Networks (CNN)
-   Conv2D
-   Max Pooling
-   ReLU activation
-   Softmax
-   Dropout
-   Flatten
-   One-Hot Encoding
-   Data Augmentation
-   Learning Rate Scheduling
-   Early Stopping
-   Model Checkpointing
-   Confusion Matrix
-   Classification Report
-   Image Classification
-   Error Analysis

## ⚠️ Notes

The notebook is based on CSV-formatted MNIST data rather than directly
loading the dataset through `keras.datasets.mnist`.

The notebook's comments/documentation mention RMSprop, but the model is
actually compiled with **Adam**. The RMSprop object is created but is
not passed to `model.compile()`.

The notebook also contains some explanatory text/results referring to
different epoch configurations. The currently configured training code
uses **100 epochs**, while the notebook comments discuss using 30 epochs
for the referenced accuracy result.

## 👨‍💻 Author

**Abdelhamid Ibrahim Abdelhamid**

Computer Science & Artificial Intelligence Graduate\
Interested in Data Analytics, Machine Learning & Deep Learning
