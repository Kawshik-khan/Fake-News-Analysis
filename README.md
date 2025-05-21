# Fake News Detection using Naive Bayes

## Project Overview

This project implements a fake news detection system using the Naive Bayes algorithm. The goal is to classify news articles as "real" or "fake" based on their text content.

## Dataset

The project uses the `fake_or_real_news.csv` dataset. This dataset contains news articles with their corresponding labels (real or fake).

## Project Steps

1.  **Data Loading and Exploration:**
    *   Load the dataset using pandas.
    *   Explore the data to understand its structure and identify any missing values.
    *   Examine the distribution of labels.

2.  **Data Preprocessing:**
    *   Remove the 'id' column as it is not relevant for classification.
    *   Separate the features (text) from the target variable (label).

3.  **Data Splitting:**
    *   Split the dataset into training and testing sets to evaluate the model's performance on unseen data.

4.  **Text Vectorization (Bag of Words):**
    *   Convert the text data into a numerical representation using the Bag of Words model (CountVectorizer). This creates a vocabulary of unique words and represents each document as a vector of word counts.

5.  **Model Training:**
    *   Train a Multinomial Naive Bayes model on the training data. Naive Bayes is a suitable algorithm for text classification tasks due to its simplicity and efficiency.

6.  **Model Evaluation:**
    *   Evaluate the trained model's performance on the testing set using metrics like accuracy, precision, recall, and F1-score.
    *   Visualize the performance using a confusion matrix to understand how well the model distinguishes between real and fake news.

7. **Model Score**
    *   Model Score are : 0.8800315706393055
    *     Precision : 
                        REAL : 0.85
                        FAKE : 0.91
          Recall :
                        REAL : 0.93
                        FAKE : 0.83
          F1-score : 
                        REAL : 0.89
                        FAKE : 0.87

8.  **Prediction on New Data:**
    *   Demonstrate how to use the trained model to predict the label of new, unseen news articles.

## Code Structure

The project is organized in a Google Colab notebook, with the following key sections:

*   Importing necessary libraries (pandas, numpy, matplotlib, seaborn, sklearn).
*   Loading and inspecting the dataset.
*   Data preprocessing steps.
*   Splitting data into training and testing sets.
*   Applying CountVectorizer for text vectorization.
*   Training the Multinomial Naive Bayes model.
*   Evaluating the model's performance.
*   Making predictions on sample news articles.

## Libraries Used

*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `matplotlib.pyplot`: For plotting.
*   `seaborn`: For enhanced visualizations.
*   `sklearn`: For machine learning algorithms and tools (model selection, feature extraction, classification metrics).

## How to Run

1.  Open the Colab notebook.
2.  Make sure the `fake_or_real_news.csv` dataset is accessible in your Colab environment.
3.  Run all the cells in the notebook sequentially.

## Results

[You can optionally include a brief summary of your model's performance here, e.g., "The model achieved an accuracy of X% on the test set."]

## Future Enhancements

*   Experiment with other text vectorization techniques (e.g., TF-IDF).
*   Explore different classification algorithms (e.g., SVM, Logistic Regression).
*   Implement more sophisticated text preprocessing steps (e.g., stemming, lemmatization, removing stop words).
*   Fine-tune the model hyperparameters for better performance.
*   Build a user interface to easily input news articles and get predictions.