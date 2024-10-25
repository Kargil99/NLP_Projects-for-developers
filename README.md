# Sentiment Analysis of Movie Reviews

## Project Overview
This project builds a model to classify movie reviews as positive or negative using various NLP techniques. The dataset used is `movie_reviews.csv`, with columns `review` and `sentiment`.

## Key Concepts
- **Bag of Words**: Represents text data in a matrix format with word frequencies.
- **N-grams**: Captures sequences of words to preserve context.
- **TF-IDF**: Weighs terms based on their importance across the dataset.
- **Word Embeddings (Word2Vec)**: Dense word vectors for capturing semantic similarities.

## Project Structure
- `sentiment_analysis.py`: Main script to preprocess data, apply NLP techniques, and evaluate different models.
- `movie_reviews.csv`: Input data with text and sentiment labels.

## Data Preprocessing
Each review is tokenized, stopwords are removed, and both stemming and lemmatization are applied to normalize the text.

## Feature Extraction Techniques
- **Bag of Words**: Simple frequency-based vectorization.
- **N-grams**: Captures unigram and bigram relationships.
- **TF-IDF**: Weights terms based on importance.
- **Word Embeddings**: Uses Word2Vec for semantic embeddings.

## Model Training
We used Naive Bayes to classify sentiment. Each feature extraction method is trained and evaluated to find the optimal approach.

## Results
| Feature Extraction | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Bag of Words       | 0.XX     | 0.XX      | 0.XX   | 0.XX     |
| N-grams            | 0.XX     | 0.XX      | 0.XX   | 0.XX     |
| TF-IDF             | 0.XX     | 0.XX      | 0.XX   | 0.XX     |
| Word2Vec           | 0.XX     | 0.XX      | 0.XX   | 0.XX     |

## Setup and Execution
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `python sentiment_analysis.py` to preprocess, train, and evaluate the models.
