# Sentiment Analysis on Movie Reviews Dataset

## Dataset Description

This dataset comprises phrases from the Rotten Tomatoes dataset, parsed into multiple phrases by the Stanford parser. The train/test split has been preserved for benchmarking purposes, but the sentences have been shuffled from their original order. Each sentence is parsed into many phrases, each with a unique `PhraseId`. Each sentence also has a unique `SentenceId`. Repeated phrases (such as common short words) are included only once in the dataset.

## File Structure

- **train.tsv.zip**: Contains phrases and their associated sentiment labels.
- **test.tsv.zip**: Contains only phrases; you must assign a sentiment label to each phrase.

### File Contents

- **train.tsv**:
  - **PhraseId**: Unique identifier for the phrase.
  - **SentenceId**: Unique identifier for the sentence.
  - **Phrase**: Text of the phrase.
  - **Sentiment**: Sentiment label (see below).

- **test.tsv**:
  - **PhraseId**: Unique identifier for the phrase.
  - **SentenceId**: Unique identifier for the sentence.
  - **Phrase**: Text of the phrase.

## Sentiment Labels

The sentiment labels are defined as follows:

- **0** - Negative
- **1** - Somewhat Negative
- **2** - Neutral
- **3** - Somewhat Positive
- **4** - Positive

## Usage Instructions

1. **Download the Dataset**:
   - You can download the dataset from the following link: [Kaggle Dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

2. **Extract the Files**:
   - Use a compatible extraction tool to unzip the `train.tsv.zip` and `test.tsv.zip` files.

3. **Data Preprocessing**:
   - It is recommended to preprocess the data appropriately before use, such as removing null values and duplicates.

4. **Model Training and Evaluation**:
   - Train your model using the training set and evaluate it using the test set.

## License

Please adhere to the terms and conditions of dataset usage.

## Contact Information

For any inquiries, please contact [Your Email or Contact Information].

