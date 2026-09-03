# Deep Learning Basics --- MNIST Classification

## 📌 Project Overview

This project is a beginner-friendly introduction to **Deep Learning
using TensorFlow/Keras**. It uses the **MNIST handwritten digit
dataset** to build and evaluate a fully connected neural network for
multi-class image classification.

The notebook demonstrates the main steps of a typical Deep Learning
workflow:

-   Loading a dataset
-   Splitting data into training and testing sets
-   Exploring image data
-   Converting data types
-   Normalizing pixel values
-   Reshaping images for a Dense network
-   Applying One-Hot Encoding to labels
-   Building a neural network with Keras
-   Compiling the model
-   Training with a validation split
-   Plotting training/validation metrics
-   Evaluating the model
-   Using a confusion matrix and classification report

## 🎯 Objective

The goal is to classify handwritten digit images into one of **10
classes (0--9)** using a feed-forward neural network.

## 📊 Dataset

The project uses the **MNIST dataset**, loaded directly through Keras:

``` python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

The notebook shows:

-   Training images: **60,000**
-   Test images: **10,000**
-   Image size: **28 × 28 pixels**
-   Number of classes: **10**

The original training data has the shape:

``` text
(60000, 28, 28)
```

After reshaping for the Dense network:

``` text
Training: (60000, 784)
Testing:  (10000, 784)
```

Each 28×28 image is therefore flattened into a vector of **784
features**.

## 🛠️ Technologies Used

-   Python 3.10.9
-   TensorFlow
-   Keras
-   NumPy
-   Matplotlib
-   Scikit-learn

## 🔄 Data Preprocessing

### 1. Convert data type

The image arrays are converted from integer values to `float32`:

``` python
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
```

### 2. Normalize pixel values

MNIST pixels range from 0 to 255. The notebook scales them to
approximately 0--1:

``` python
x_train /= 255
x_test /= 255
```

### 3. Flatten the images

Because the model uses Dense layers rather than convolutional layers,
each 28×28 image is flattened:

``` python
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
```

### 4. One-Hot Encoding

The labels are converted into 10-dimensional one-hot vectors:

``` python
from keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
```

The resulting label shapes are:

``` text
y_train: (60000, 10)
y_test:  (10000, 10)
```

## 🧠 Model Architecture

The notebook builds a Keras `Sequential` neural network with the
following architecture:

``` text
Input: 784 features
        ↓
Dense: 128 neurons + ReLU
        ↓
Dense: 256 neurons + ReLU
        ↓
Dense: 256 neurons + ReLU
        ↓
Dense: 64 neurons + ReLU
        ↓
Dense: 10 neurons + Softmax
```

### Architecture Details

  Layer           Units Activation
  ------------- ------- ------------
  Input/Dense       128 ReLU
  Dense             256 ReLU
  Dense             256 ReLU
  Dense              64 ReLU
  Output             10 Softmax

The model summary in the notebook reports:

-   **Total parameters:** 216,394
-   **Trainable parameters:** 216,394
-   **Non-trainable parameters:** 0

## ⚙️ Model Compilation

The model is compiled using:

``` python
model.compile(
    loss="categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)
```

### Configuration

-   **Loss:** Categorical Crossentropy
-   **Optimizer:** SGD
-   **Metric:** Accuracy

`categorical_crossentropy` is appropriate here because the labels are
represented using One-Hot Encoding and the output layer uses Softmax for
10-class classification.

## 🏋️ Model Training

The notebook is configured to train for:

``` text
Epochs: 100
Batch size: 128
Validation split: 10%
```

Training is performed with:

``` python
model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=128
)
```

The notebook also includes a note to run the training on a GPU.

## 📈 Training Visualization

The training history is used to plot:

-   Training accuracy
-   Validation accuracy
-   Training loss
-   Validation loss

This helps monitor how the model behaves during training.

## 🧪 Model Evaluation

The notebook includes evaluation on the test set:

``` python
model.evaluate(x_test, y_test)
```

It also attempts to generate:

-   Confusion Matrix
-   Classification Report

These metrics provide a more detailed view of classification performance
than accuracy alone.

## ⚠️ Notes About the Current Notebook

The README describes the code that is currently present in the uploaded
notebook. A few cells appear to need fixing before the complete
evaluation section can run successfully:

1.  `test_acc` is printed but is not defined in the notebook.
2.  `y_pred` is used for the confusion matrix and classification report
    but is not defined in the notebook.
3.  The `classification_report` line contains an extra closing
    parenthesis.

For example, predictions would need to be generated before using
`y_pred`:

``` python
y_pred = model.predict(x_test)
```

And `test_acc` should be assigned from the result of `model.evaluate()`
before printing it.

These issues do **not** change the main learning objective of the
notebook, but they should be corrected if the repository is intended to
contain a fully executable project.

## 📁 Project Structure

``` text
.
├── basics deep learning.ipynb
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install the required libraries

``` bash
pip install tensorflow keras numpy matplotlib scikit-learn
```

### 3. Open the notebook

You can run the notebook using:

-   Jupyter Notebook
-   JupyterLab
-   Google Colab
-   VS Code with the Jupyter extension

### 4. Run the cells in order

Run the preprocessing, model-building, compilation, and training cells
sequentially.

> **Tip:** Training for 100 epochs can take some time. A GPU can
> significantly improve training speed.

## 📚 Key Deep Learning Concepts Covered

This notebook is useful for understanding the foundations of:

-   Neural Networks
-   Dense / Fully Connected Layers
-   ReLU Activation
-   Softmax Activation
-   One-Hot Encoding
-   Normalization
-   Flattening Image Data
-   Categorical Crossentropy
-   SGD Optimizer
-   Epochs
-   Batch Size
-   Validation Split
-   Model Evaluation
-   Confusion Matrix
-   Classification Report

## 💡 Learning Outcome

After completing this notebook, you should have a basic understanding of
how to take image data, preprocess it, convert labels into a suitable
format, build a neural network with Keras, train it, and evaluate its
classification performance.

This project is a foundation for moving from basic **ANN/Dense
networks** toward more specialized architectures such as **CNNs**, which
are better suited to image data.

## 👨‍💻 Author

**Abdelhamid Ibrahim Abdelhamid**

Computer Science & Artificial Intelligence Graduate\
Interested in Data Science, Machine Learning & Deep Learning
