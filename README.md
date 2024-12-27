# Diabetes Classification Project Documentation

## Project Overview

This project aims to classify individuals as diabetic (1) or non-diabetic (0) based on various health indicators. The Pima Indians Diabetes dataset is used for this purpose, containing features like glucose level, BMI, and age, among others. The target variable is "Outcome," indicating whether a person has diabetes.

## Goals

- Clean and Fix the data.
- Understand the dataset structure and perform exploratory data analysis (EDA).
- Preprocess the data for effective modeling.
- Train various classification models and evaluate them to pick up the best one.
- Optimize hyperparameters using RandomizedSearchCV.

---

## 1. Importing Libraries

The following libraries are imported:

- **Data Manipulation and Analysis**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Model Training and Evaluation**: `sklearn` (for models like Logistic Regression, Decision Trees, etc.), `xgboost` (for gradient boosting), and `sklearn.metrics` (for performance evaluation)
- **Preprocessing**: `StandardScaler`, `LabelEncoder`
- **Hyperparameter Tuning**: `RandomizedSearchCV`

Additionally, warnings are suppressed to avoid unnecessary clutter in the output.

---

## 2. Dataset Loading and Exploration

The dataset is loaded into a DataFrame using `pandas`. Key steps:

- **Understanding the Dataset**: Inspect the first few rows using `.head()` and summarize statistics using `.describe()`.
- **Missing Values**: Check for null values using `.isnull().sum()`.
- **Class Distribution**: Analyze the distribution of the target variable (`Outcome`).

---

## 3. Data Preprocessing

Preprocessing steps include:

1. **Handling Missing Values**: If present, impute missing data using appropriate strategies (e.g., mean for numerical columns).
2. **Feature Scaling**: Apply `StandardScaler` to normalize features like glucose and BMI.
3. **Encoding Categorical Data**: Use `LabelEncoder` if any categorical features are present.
4. **Feature Engineering**: Create new features or group existing ones to improve model performance.

---

## 4. Exploratory Data Analysis (EDA)

Visualization techniques used:

- **Distribution Plots**: Understand feature distributions using histograms and KDE plots.
- **Correlation Matrix**: Identify relationships between features using a heatmap.
- **Boxplots**: Detect outliers and visualize feature distributions across classes.

---

## 5. Model Training

Various classifiers are trained:

1. Logistic Regression
2. Decision Trees
3. Random Forest
4. Gradient Boosting (including XGBoost)
5. Support Vector Machines (SVM)
6. k-Nearest Neighbors (k-NN)
7. Naïve Bayes
8. AdaBoost

Models are evaluated using metrics such as:

- **Accuracy**
- **Recall** (critical for medical diagnosis)

---

## 6. Hyperparameter Tuning

RandomizedSearchCV is used for optimizing hyperparameters. Example:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(random_state=0),
    param_distributions=param_grid,
    n_iter=50,
    scoring='recall',
    cv=5,
    verbose=1,
    random_state=0,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

---

## 7. Model Evaluation

The best model is evaluated on the test set using:

- **Recall**: Measures the proportion of actual positives correctly identified.
- **Accuracy**: Provides overall model correctness.

Example code:

```python
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
recall = recall_score(y_test, y_pred)
print(f'Recall: {recall:.4f}')
```

---

## 8. Results and Observations

- Summarize the best-performing model and its hyperparameters.
- Provide recall and accuracy values for the test set.

---
