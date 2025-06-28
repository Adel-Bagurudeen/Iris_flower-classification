

# Iris Flower Classification 

## Overview
This machine learning project focuses on classifying iris flowers into three distinct species—*Iris setosa*, *Iris versicolor*, and *Iris virginica*—based on their sepal and petal measurements. The project employs the Logistic Regression algorithm, a robust supervised learning technique, to construct a predictive model. The primary objective is to accurately predict the species of an iris flower using its morphological features.

## Project Objectives
- Develop a classification model using Logistic Regression.
- Utilize the Iris dataset, comprising 150 samples, for training and evaluation.
- Assess model performance using accuracy as the key metric.
- Optionally visualize the dataset to explore class distributions and feature relationships.

## Dataset Overview
The Iris dataset, a cornerstone in machine learning research, contains 150 samples with the following attributes:

| Feature | Description |
|---------|-------------|
| Sepal Length (cm) | Length of the sepal |
| Sepal Width (cm) | Width of the sepal |
| Petal Length (cm) | Length of the petal |
| Petal Width (cm) | Width of the petal |
| Species | Target class (*Setosa*, *Versicolor*, *Virginica*) |

The dataset is balanced, with 50 samples per species, labeled as:
- 0: *Iris setosa*
- 1: *Iris versicolor*
- 2: *Iris virginica*

## Tools and Libraries
This project is implemented in Python within a Jupyter Notebook environment (e.g., Google Colab). The following libraries are utilized:

| Library | Purpose |
|---------|---------|
| pandas | Data handling and manipulation using DataFrames |
| numpy | Numerical computations and array operations |
| matplotlib.pyplot | Visualization of data distributions (optional) |
| sklearn.datasets | Loading the Iris dataset |
| sklearn.model_selection | Splitting data into training and testing sets |
| sklearn.linear_model | Implementing the Logistic Regression algorithm |
| sklearn.metrics | Evaluating model performance via accuracy |

## Implementation Steps
1. **Library Imports**: Load essential Python libraries for data processing, modeling, and evaluation.
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score
   ```

2. **Data Loading and Preparation**: Import the Iris dataset and organize it into features (X) and target labels (y).
3. **Data Splitting**: Divide the dataset into training (e.g., 80%) and testing (e.g., 20%) sets using `train_test_split`.
4. **Model Training**: Initialize and train the Logistic Regression model on the training data.
5. **Prediction and Evaluation**: Generate predictions on the test set and compute the accuracy score.
6. **Visualization (Optional)**: Create scatter plots or pair plots to visualize feature distributions across species.

## Results
- **Model Accuracy**: Achieved an accuracy of 100% on the test set, demonstrating perfect predictive performance.
- **Sample Prediction**: The model successfully classifies new iris samples based on their features with no errors.



