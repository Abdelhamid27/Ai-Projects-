# 📰 Fake News Classification – NLP Experiments

A Natural Language Processing (NLP) project for **Fake News Detection** using the **WELFake Dataset**.

The project uses the same dataset while experimenting with different text representations and classification approaches across four notebooks:

1. `fake-news-classification.ipynb`
2. `fake-news-classification_text_columns.ipynb`
3. `fake-news using n-gram.ipynb`
4. `fake-news using n-gram_text_column.ipynb`

The goal is to investigate how different ways of representing news content can be used to distinguish between **Fake** and **Real** news.

---

## 🎯 Project Objective

The objective is to build machine learning models capable of classifying a news article into one of two classes:

- **0 → Fake News**
- **1 → Real News**

The project focuses on an NLP workflow starting from raw news data, followed by data cleaning, text preprocessing, feature extraction, model training, and evaluation.

---

## 📊 Dataset

The project uses the **WELFake Dataset** stored in:

```text
WELFake_Dataset.csv
```

The dataset contains news information with the following main columns:

| Column | Description |
|---|---|
| `Unnamed: 0` | Original dataset index |
| `title` | News headline |
| `text` | News article content |
| `label` | Target class: `0 = Fake`, `1 = Real` |

### Dataset Preparation

The notebook initially creates a balanced sample by selecting:

- **5,000 samples from label 0**
- **5,000 samples from label 1**

Total:

```text
10,000 samples
```

After removing missing values:

```text
9,913 samples
```

The unused `Unnamed: 0` column is then removed.

The resulting dataset contains:

```text
title
text
label
```

The class distribution is treated as balanced in the notebook.

---

# 🔄 Project Experiments

The four notebooks represent different experiments on the same Fake News classification task.

## 1. `fake-news-classification.ipynb`

This notebook represents the main classification experiment.

### Text Used

The recorded preprocessing pipeline works with the **news title**.

### Text Preprocessing

The title text is processed using:

- Regular Expression cleaning
- Lowercasing
- Tokenization through `split()`
- English stopword removal
- Lemmatization using `WordNetLemmatizer`

The notebook also initializes a `PorterStemmer`, although the recorded preprocessing pipeline applies lemmatization rather than stemming.

Example of the cleaned text:

```text
new jersey judge order naming bridgegate scandal co conspirator
```

### TF-IDF Feature Extraction

The cleaned corpus is transformed into numerical features using:

```python
TfidfVectorizer(max_features=5000)
```

Result:

```text
Samples: 9,913
Features: 5,000
```

### Train/Test Split

The data is divided using:

```python
train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The recorded test set contains:

```text
1,987 samples
```

---

# 🤖 Machine Learning Models

Several traditional machine learning algorithms were tested on the TF-IDF representation.

### Individual Models

- Random Forest
- Multinomial Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Gradient Boosting
- AdaBoost
- XGBoost

### Ensemble Models

The project also experiments with:

- Voting Classifier
- Stacking Classifier

---

# 📈 Recorded Results

The following results are recorded in `fake-news-classification.ipynb`.

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| **Stacking Classifier** | **88.27%** | **88.39%** | **87.94%** | **88.17%** |
| SVM | 87.17% | 86.17% | 88.35% | 87.24% |
| Logistic Regression | 87.12% | 86.22% | 88.15% | 87.17% |
| Random Forest | 86.01% | 84.59% | 87.84% | 86.18% |
| Naive Bayes | 85.56% | 86.76% | 83.69% | 85.20% |
| XGBoost | 84.55% | 82.20% | 87.94% | 84.97% |
| Voting Classifier | 85.00% | 80.19% | 92.71% | 86.00% |
| Gradient Boosting | 82.13% | 77.96% | 89.26% | 83.23% |
| Decision Tree | 81.13% | 81.61% | 80.04% | 80.82% |
| AdaBoost | 77.35% | 70.14% | 94.73% | 80.60% |
| KNN | 50.28% | 49.97% | 99.90% | 66.62% |

### 🏆 Best Recorded Model

The **Stacking Classifier** achieved the highest recorded accuracy:

```text
Accuracy : 88.27%
Precision: 88.39%
Recall   : 87.94%
F1 Score : 88.17%
```

---

# 🔬 2. `fake-news-classification_text_columns.ipynb`

This notebook is one of the project's alternative experiments and focuses on using the available **text columns** of the news data rather than treating the headline as the only text source.

The purpose of this experiment is to investigate whether incorporating additional textual information can improve the classification of Fake and Real news.

---

# 🔤 3. `fake-news using n-gram.ipynb`

This notebook explores **N-gram based text representation** for Fake News classification.

Instead of relying only on individual words, N-grams can represent sequences of words and therefore capture short word combinations and local textual patterns.

Examples:

```text
Unigram → fake
Bigram  → fake news
Trigram → fake news detection
```

This experiment is intended to compare N-gram based features with the standard text representation approach.

---

# 🔤 4. `fake-news using n-gram_text_column.ipynb`

This notebook combines the project's two experimental directions:

- **N-gram based text representation**
- **Multiple text columns**

The purpose is to investigate whether using richer textual information together with word-sequence features can provide a stronger representation for Fake News classification.

---

# 🧹 NLP Preprocessing Pipeline

The main NLP workflow can be summarized as:

```text
Raw News Data
      ↓
Select Balanced Classes
      ↓
Handle Missing Values
      ↓
Remove Unused Column
      ↓
Select Text
      ↓
Regex Cleaning
      ↓
Lowercasing
      ↓
Tokenization
      ↓
Remove Stopwords
      ↓
Lemmatization
      ↓
TF-IDF / N-gram Representation
      ↓
Train / Test Split
      ↓
Machine Learning Models
      ↓
Evaluation
```

---

# 📏 Evaluation Metrics

The project evaluates the classification models using:

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many samples predicted as a particular positive class were actually positive.

### Recall

Measures how many actual positive samples were correctly detected.

### F1 Score

The harmonic mean of Precision and Recall.

### Confusion Matrix

Used to visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

# 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- XGBoost
- Google Colab

### NLP Tools

- NLTK Stopwords
- WordNet Lemmatizer
- Porter Stemmer
- Regular Expressions
- TF-IDF Vectorization
- N-gram feature representation

### Machine Learning

- Random Forest
- Naive Bayes
- Logistic Regression
- SVM
- KNN
- Decision Tree
- Gradient Boosting
- AdaBoost
- XGBoost
- Voting Ensemble
- Stacking Ensemble

---

# 📁 Project Structure

```text
Fake News Classification/
│
├── WELFake_Dataset.csv
│
├── fake-news-classification.ipynb
├── fake-news-classification_text_columns.ipynb
├── fake-news using n-gram.ipynb
├── fake-news using n-gram_text_column.ipynb
│
└── README.md
```

---

# 🚀 How to Run

The notebooks were developed for **Google Colab**.

### 1. Open a notebook

Open any of the four `.ipynb` files in Google Colab.

### 2. Prepare the dataset

Make sure:

```text
WELFake_Dataset.csv
```

is available in the notebook's working directory.

### 3. Install/Import Dependencies

The notebooks use Python NLP and machine learning libraries including NLTK, Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib, and XGBoost.

### 4. Run the cells sequentially

The workflow covers:

1. Dataset loading
2. Data exploration
3. Missing-value handling
4. Text preprocessing
5. Feature extraction
6. Train/test splitting
7. Model training
8. Model evaluation
9. Confusion matrix visualization

---

# 📌 Key Takeaways

- The project applies **NLP techniques to Fake News Detection**.
- The dataset is balanced by sampling 5,000 records from each class before preprocessing.
- Missing values are removed before modeling.
- Text preprocessing includes cleaning, lowercasing, tokenization, stopword removal, and lemmatization.
- TF-IDF is used to convert text into numerical features in the main classification experiment.
- Multiple traditional machine learning algorithms are compared.
- Ensemble learning is also explored through Voting and Stacking.
- The **Stacking Classifier achieved the highest recorded accuracy of 88.27%** in the main notebook.
- The other notebooks extend the same project by experimenting with **text columns and N-gram representations**.

---

## ⚠️ Note

The reported metrics in this README are the recorded results from `fake-news-classification.ipynb`.

The four notebooks are presented as a unified NLP project because they use the same Fake News classification problem and explore different text/feature representations. Exact model metrics for the other three notebooks should be added from their recorded outputs if they are to be reported separately.

---

## 📄 Dataset Reference

**WELFake Dataset**

The dataset is used for educational and machine learning experimentation in Fake News classification.
