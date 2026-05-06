---
title: "Text Classification for the LING 539 Kaggle Competition"
author: "Anitha Karni"
date: "2026-04-30"
---

## Introduction

For my term project, I participated in the LING 539 class Kaggle competition. The goal was
to build a text classification system using the provided training data and generate predicted
labels for an unseen test set. My approach focused on traditional NLP methods covered in the
course, especially TF-IDF feature extraction and linear classifiers.

## Data and Task

The competition data was distributed through a private Kaggle competition. The training set
contained text examples and gold labels, while the test set contained text examples without
labels. The required output format was a CSV file with the columns `ID` and `LABEL`.

The dataset schema used in my project was:

- `train.csv`: `ID`, `TEXT`, `LABEL`
- `test.csv`: `ID`, `TEXT`
- `sample_submission.csv`: `ID`, `LABEL`

## Preprocessing

I applied a lightweight but effective preprocessing pipeline to prepare the text for
traditional machine learning models:

1. HTML entities were decoded and HTML tags were removed.
2. Text was lowercased.
3. Non-alphanumeric characters were removed.
4. Extra whitespace was normalized.
5. Porter stemming was applied.

These steps helped reduce noise while keeping the pipeline interpretable and reproducible.

## Feature Extraction

I represented each document with TF-IDF features. I used unigram and bigram features so the
model could capture both individual words and short phrases. TF-IDF is a strong baseline for
text classification because it highlights terms that are informative for a document while
down-weighting very common terms.

## Models

I compared two course-relevant classification models:

1. Logistic Regression
2. Multinomial Naive Bayes

Both models were trained with the same TF-IDF representation so the comparison focused on the
classifier rather than changing multiple variables at once.

## Validation and Model Selection

Before generating Kaggle predictions, I split the training data into an 80/20 train and
validation partition using a stratified split. I evaluated each model with accuracy,
precision, recall, and weighted F1-score.

From my saved validation run, Logistic Regression performed best:

- Accuracy: `0.9285`
- Weighted precision: `0.9278`
- Weighted recall: `0.9285`
- Weighted F1-score: `0.9280`

Because Logistic Regression gave the strongest weighted F1-score, I selected it as my final
model and retrained it on the full training dataset.

## Kaggle Submission

After choosing the final model, I generated predictions for the test set and saved them in
the required Kaggle submission format. I then uploaded the resulting `submission.csv` file to
the private competition page.

Kaggle leaderboard result:

- Public leaderboard score: `0.92696`

## Reflection

This project showed that traditional NLP methods can still provide strong results on text
classification tasks. One advantage of the TF-IDF plus Logistic Regression pipeline is that
it is straightforward to implement, fast to train, and relatively easy to interpret. The
main limitation is that it relies on surface-level lexical features and cannot capture deep
context as effectively as transformer-based models.

If I continued this project, I would try a few next steps:

- Compare against Linear SVM
- Tune TF-IDF hyperparameters further
- Perform more detailed error analysis by label
- Test a transformer-based classifier as a stronger neural baseline

## Conclusion

This term project helped me work through the full NLP workflow: preparing data, engineering
features, training and evaluating classifiers, generating a Kaggle submission, and explaining
the design choices clearly in writing. It also reinforced how a well-tuned traditional model
can be a strong baseline for real text classification tasks.
