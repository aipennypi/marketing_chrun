Waze Dataset Churn Prediction:<br>
Machine Learning Approach : waze_churn.py<br>
Deep Learning Approach : ChurnAnalysis.py<br>

This project is focused on predicting user churn using the Waze dataset. It implements exploratory data analysis (EDA), feature engineering, and machine learning modeling with Random Forest and XGBoost classifiers. The project also compares the performance of these models using metrics such as F1-score, precision, recall, and accuracy.

Features
Data Preprocessing:

Handles missing values and performs feature transformation.
Encodes categorical variables and drops irrelevant features.
Stratified train-test splitting for imbalanced data.
Exploratory Data Analysis (EDA):

Provides insights into features grouped by target labels.
Visualizes data distributions and calculates feature averages for different churn statuses.
Machine Learning Models:

Random Forest:
Performs hyperparameter tuning using GridSearchCV.
Validates performance using custom cross-validation splits.

XGBoost:
Implements hyperparameter optimization with GridSearchCV.

Sequencial deep learning model:
two hidden layer 
Adam optimizer

Model Evaluation:
Computes F1-score, precision, recall, and accuracy.
Displays confusion matrices for further analysis.
Compares feature importances between models.
Visualization:

Visualizes feature importances for Random Forest and XGBoost models.
Creates bar plots for top contributing features.

Results
Model Performance Metrics:

Evaluated and compared F1-score, recall, precision, and accuracy for both models.
Exported results to waze_Results.csv.
Feature Importance:

Displayed the top 10 most influential features for predicting user churn.
