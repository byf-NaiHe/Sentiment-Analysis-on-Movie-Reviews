# Sentiment Analysis on Movie Reviews

## Description

This project focuses on sentiment analysis of movie reviews using the Rotten Tomatoes dataset. The dataset contains phrases from movie reviews labeled with fine-grained sentiment scores on a scale of five values: **negative**, **somewhat negative**, **neutral**, **somewhat positive**, and **positive**. The task is challenging due to obstacles such as sentence negation, sarcasm, terseness, and language ambiguity.

The Rotten Tomatoes dataset was originally collected by Pang and Lee [1] and later enhanced by Socher et al. [2] using Amazon's Mechanical Turk to create a sentiment treebank. This project benchmarks various machine learning models and word embedding techniques to tackle this sentiment analysis problem.

[1] Pang and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In ACL, pages 115–124.

[2] Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank, Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Chris Manning, Andrew Ng and Chris Potts. Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).

---

## Project Overview

This project explores the effectiveness of **five machine learning models** and **different word embedding techniques** for sentiment analysis on the Rotten Tomatoes dataset. The models and techniques are compared to determine their performance in predicting sentiment labels.

### Models Used:
1. **Random Forest**
2. **LSTM (Long Short-Term Memory)**
3. **BERT (Bidirectional Encoder Representations from Transformers)**
4. **XGBoost**
5. **Simple Neural Network (Baseline)**

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

### 2. Model Implementation
- **Random Forest**: A traditional ensemble method for classification.
- **LSTM**: A recurrent neural network model for sequence data.
- **BERT**: A transformer-based model for contextual embeddings.
- **XGBoost**: A gradient boosting algorithm for structured data.
- **Simple Neural Network**: A baseline model for comparison.

### 3. Evaluation Metrics
- **Accuracy**: Percentage of correctly classified phrases.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: Visualization of model performance across classes.

---

## Results

| Model         | Word Embedding | Accuracy | F1-Score | Notes                          |
|---------------|----------------|----------|----------|--------------------------------|
| Random Forest | TF-IDF         | 0.62     | 0.60     | Performs well with simple text features. |
| LSTM          | Word2Vec       | 0.68     | 0.66     | Captures sequential dependencies.        |
| BERT          | BERT Embeddings| 0.75     | 0.74     | State-of-the-art performance.            |
| XGBoost       | TF-IDF         | 0.64     | 0.62     | Good balance of speed and accuracy.      |
| Simple NN     | GloVe          | 0.58     | 0.56     | Baseline model for comparison.           |

---

## Key Findings
- **BERT** outperformed all other models due to its ability to capture contextual information.
- **LSTM** performed well but was slower to train compared to BERT.
- **Random Forest** and **XGBoost** were effective with simpler embeddings like TF-IDF.
- **Word2Vec** and **GloVe** provided moderate improvements over TF-IDF for deep learning models.

---

## How to Use This Repository

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/sentiment-analysis-movie-reviews.git
cd sentiment-analysis-movie-reviews
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download the Rotten Tomatoes dataset from [Kaggle](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) and place it in the `data/` directory.

### 4. Run the Code
- **Preprocessing**: Run `preprocess.py` to clean and prepare the data.
- **Training**: Use the following scripts to train the models:
  - `train_random_forest.py`
  - `train_lstm.py`
  - `train_bert.py`
  - `train_xgboost.py`
  - `train_simple_nn.py`
- **Evaluation**: Run `evaluate.py` to generate performance metrics.

### 5. Generate Predictions

```bash
Use the trained models to generate predictions on the test.tsv set
```

---

## Directory Structure

```plaintext
sentiment-analysis-movie-reviews/
├── data/                   # Dataset files
├── models/                 # Saved models
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

## Contact

For questions or feedback, please open an issue or contact [2200012137@stu.pku.edu.cn](mailto:2200012137@stu.pku.edu.cn).

---

Happy coding! 🎬🍿
