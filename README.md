# Sentiment Analysis on Movie Reviews

## Description

<p>&nbsp;&nbsp;This project focuses on sentiment analysis of movie reviews using the Rotten Tomatoes dataset. The dataset contains phrases from movie reviews labeled with fine-grained sentiment scores on a scale of five values: <strong>negative</strong>, <strong>somewhat negative</strong>, <strong>neutral</strong>, <strong>somewhat positive</strong>, and <strong>positive</strong>. The task is challenging due to obstacles such as sentence negation, sarcasm, terseness, and language ambiguity.</p>

<p>&nbsp;&nbsp;The Rotten Tomatoes dataset was originally collected by Pang and Lee [1] and later enhanced by Socher et al. [2] using Amazon's Mechanical Turk to create a sentiment treebank. This project benchmarks various machine learning models and word embedding techniques to tackle this sentiment analysis problem.</p>


[1] Pang and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In ACL, pages 115–124.

[2] Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank, Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Chris Manning, Andrew Ng and Chris Potts. Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).

---

## Project Overview

<p>&nbsp;&nbsp;This project explores the effectiveness of <strong>five machine learning models</strong> and <strong>different word embedding techniques</strong> for sentiment analysis on the Rotten Tomatoes dataset. The models and techniques are compared to determine their performance in predicting sentiment labels.</p>

### Models Used:
1. **Multilayer Perceptron**
2. **XGBoost**
3. **Random Forest**
4. **LSTM (Long Short-Term Memory)**
5. **BERT (Bidirectional Encoder Representations from Transformers)**

### Word Embedding Techniques:
- **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Word2Vec**
- **BERT Embeddings**

---

## Dataset

The dataset consists of:
- **Train Data**: Phrases and their corresponding sentiment labels.
- **Test Data**: Phrases for which sentiment labels need to be predicted.

Each phrase is labeled with one of the following sentiment classes:
- 0: Negative
- 1: Somewhat Negative
- 2: Neutral
- 3: Somewhat Positive
- 4: Positive

---

## Methodology

### 1. Data Preprocessing
- **Text Cleaning**: Lowercasing, removing punctuation, and handling stopwords.
- **Tokenization**: Splitting text into individual words or tokens.
- **Word Embeddings**: Converting text into numerical representations using:
  - **TF-IDF**: For traditional machine learning models.
  - **Word2Vec**: For deep learning models.
  - **BERT**: For state-of-the-art contextual embeddings.

### 2. Model Construction
- **Multilayer Perceptron**: A baseline model.
- **XGBoost**: A gradient boosting algorithm for structured data.
- **Random Forest**: A traditional ensemble method for classification.
- **LSTM**: A recurrent neural network model for sequence data.
- **BERT**: A transformer-based model for contextual embeddings.

### 3. Evaluation Metrics
- **Accuracy**: Percentage of correctly classified phrases.
- **Confusion Matrix**: Visualization of model performance across classes.

---

## Key Findings
- **BERT** outperformed all other models due to its ability to capture contextual information.
- **LSTM** performed well but was slower to train compared to BERT.
- **Random Forest** and **XGBoost** were effective with simpler embeddings like TF-IDF.
- **Word2Vec** and **GloVe** provided moderate improvements over TF-IDF for deep learning models.

---

## Directory Structure

```plaintext
sentiment-analysis-movie-reviews/
├── data/                   # Dataset files
├── codes/                  # different models codes
├── results/                # Performance metrics and plots
├── requirements.txt        # List of dependencies
└── README.md               # Project documentation
```

---

## Future Work

- Experiment with other transformer-based models like RoBERTa or DistilBERT.
- Incorporate additional features such as part-of-speech tags or sentiment lexicons.
- Perform hyperparameter tuning for better model performance.

---

## Acknowledgments

- Kaggle for hosting the competition and providing the dataset.
- The authors of the Rotten Tomatoes dataset and sentiment treebank.
- The open-source community for providing libraries like PyTorch, TensorFlow, and Hugging Face Transformers.

---

Happy coding! 🎬🍿
