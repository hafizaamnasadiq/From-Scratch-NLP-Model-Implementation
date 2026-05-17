# From-Scratch-NLP-Model-Implementation
NLP text classification approach from scratch in Python. The goal is to explore different modeling assumptions about language.

## Character-Level N-Gram Naive Bayes Text Classifier
from-scratch text classification model using a Character-Level N-Gram Naive Bayes approach, without any external machine learning libraries.

It classifies text based on character sequence patterns rather than words.
"free money"
→ ["_fr", "fre", "ree", "ee ", "e m", " mo", "mon", "one", "ney", "ey_"]

The model computes:
P(class | text) ∝ P(class) × ∏ P(ngram | class)

Using log probabilities:
log P(class) + Σ log P(ngram | class)
📦 Features
-Pure Python implementation (no ML libraries)
-Character-level n-gram extraction
-Naive Bayes classifier
-Laplace smoothing
-Batch prediction support
-Evaluation (accuracy, predictions)
-Train/test split support
-Optional multi-epoch training simulation

Project/
│
├── nGram.py              # Main model implementation
├── loadData.py          # Dataset loader (custom)
├── dataset.csv          # Input data (text, label)
└── README.md            # Documentation

⚠️ Limitations
-Sensitive to small datasets
-Not context-aware (unlike transformers)
-Can bias toward dominant class if data is imbalanced
-No semantic understanding
