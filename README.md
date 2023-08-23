# Customer Churn Prediction with Machine Learning

## Introduction
In this project, I tackled the challenge of predicting customer churn for a telecommunications company. Churn rate, the rate at which customers discontinue using a product or service, is a critical metric that impacts a company's revenue and growth. The objective was to develop machine learning models that can accurately predict whether a customer is likely to churn, enabling the company to take proactive actions to retain customers.

## Project Steps:

### 1) Exploratory Data Analysis (EDA)
- I began with exploratory data analysis to understand the dataset's characteristics and relationships. Key insights from EDA include identifying class imbalance in the target variable (churn), visualizing the distribution of tenure and its relationship with churn, and exploring the impact of different contract types on churn.

### 2) Modeling
I experimented with several machine learning algorithms to predict customer churn. The models used were:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Support Vector Classifier (SVC)

To address class imbalance, I utilized the Synthetic Minority Over-sampling Technique (SMOTE) to balance the training set. This technique helps the models better handle imbalanced data.

### 3) Model Evaluation
I evaluated the models' performance using accuracy and other relevant metrics. Key evaluation steps include:

ROC Curves: Visualizing the trade-off between true positive rate and false positive rate.
Confusion Matrix: Understanding the distribution of true positives, true negatives, false positives, and false negatives.
Classification Report: Providing precision, recall, F1-score, and support for each class.
Precision-Recall Curve: Highlighting the precision-recall trade-off for different threshold values.

### 4) Hyperparameter Tuning
Hyperparameter tuning was conducted using Randomized Search and Grid Search techniques for models like Random Forest and Logistic Regression. Tuning helps optimize model performance by finding the best combination of hyperparameters.

## Conclusion
This project aims to provide businesses with a tool for predicting customer churn using machine learning. Using historical customer data and employing machine learning algorithms, the tool will assist in identifying customers who are likely to churn. 

## Future work
This is a work in progress. I will work on finding feature importance and interpret them in order to understand what feature(s) has the most influence in customer churn.
