import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
path = '/Users/amos/PycharmProjects/pythonProject2/'
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import time

pd.set_option('display.max_columns', None)


# Read the data
df=pd.read_csv("waze_dataset.csv")

print (df.head())
print ("info:\n",df.info())
print ("more information:\n",df.describe())
print ("data type:\n",df.dtypes)
print ("Column:\n",df.columns)
print ("row * column:\n",df.shape)

# checking the imbalance
print ("Label:\n",df["label"].unique())
print(df["label"].value_counts(normalize=True))
print ("devices:",df["device"].unique())

#checking the missing value
df_drop=df.dropna()
print (df.isnull().sum())
print (df_drop.isnull().sum())
print ("df_drop:\n",df_drop.shape)
# feature transformation
df_drop['label']=df_drop['label'].replace({'retained': 0, 'churned': 1})
df_drop['device']=df_drop['device'].replace({'Android': 0, 'iPhone': 1})
print ("Label:\n",df_drop["label"].unique())
print ("devices:",df_drop["device"].unique())

# EDA
print (df_drop[['label', 'n_days_after_onboarding']].groupby('label').mean())
print (df_drop[['label', 'activity_days']].groupby('label').mean())
print (df_drop[['label', 'driven_km_drives']].groupby('label').mean())
print (df_drop[['label', 'duration_minutes_drives']].groupby('label').mean())
print (df_drop[['label', 'total_sessions']].groupby('label').mean())
print (df_drop[['label', 'total_navigations_fav1']].groupby('label').mean())
print (df_drop[['label', 'total_navigations_fav2']].groupby('label').mean())
# drop this one
print (df_drop[['label', 'device']].groupby('label').mean())
df_drop=df_drop.drop(columns=['ID','device'])
print (df_drop.columns)

#split
# Split into train and test sets
X=df_drop.copy()
X=X.drop(columns='label')
y=df_drop['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)
print ("X:\n",X.columns)
print ("X:\n",X.shape)
print ("y:",y.shape)

# Modeling
cv_params = {'max_depth': [2,3,4,5,6, None],
             'min_samples_leaf': [1,2,3,4],
             'min_samples_split': [2,3,4,5],
             'max_features': [2,3,4,5],
             'n_estimators': [75, 100, 125]
             }

scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Create separate validation data
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25,
                                            stratify=y_train, random_state=10)
# Create list of split indices
# 0 : validate -1 training data set
split_index = [0 if x in X_val.index else -1 for x in X_train.index]
from sklearn.model_selection import PredefinedSplit

rf = RandomForestClassifier(random_state=10)

custom_split = PredefinedSplit(split_index)
# CV the number of folde in cross vaildation : use custom_split to control
# refit which metric for tunning model
rf_t0 = time.time()
rf_val = GridSearchCV(rf, cv_params, scoring=scoring, cv=custom_split, refit='f1')
rf_val.fit(X_train, y_train)
# use pickl
with open(path+'waze.pickle', 'wb') as to_write:
    pickle.dump(rf_val, to_write)
rf_time = time.time()-rf_t0
print ("time_elapse:",rf_time)
print ("rf_val.best_params_forest:")
print (rf_val.best_params_)


def make_results(model_name, model_object):
    '''
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.
    '''

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean f1 score)
    # Change here if the score balaceline changed
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame()
    table = table.append({'Model': model_name,
                          'F1': f1,
                          'Recall': recall,
                          'Precision': precision,
                          'Accuracy': accuracy
                          },
                         ignore_index=True
                         )

    return table


rf_val_results = make_results('Random Forest Validated', rf_val)
# print (rf_val_results )
rf_val_results.to_csv("waze_Results.csv")

# Predict on test data

forest_cv_preds = rf_val.predict(X_test)
print("random forest CV:\n",rf_val_results)
print("random forest Result:")
print ("="*300)
print('F1 score final RandomForest model: ', f1_score(y_test, forest_cv_preds))
print('Recall score final RandomForest model: ', recall_score(y_test, forest_cv_preds))
print('Precision score final RandomForest model: ', precision_score(y_test, forest_cv_preds))
print('Accuracy score final RandomForest model: ', accuracy_score(y_test, forest_cv_preds))


# GBM Modeling
xgb = XGBClassifier(objective='binary:logistic', random_state=0)
cv_params = {'max_depth': [4,5,6,7,8],
             'min_child_weight': [1,2,3,4,5],
             'learning_rate': [0.1, 0.2, 0.3],
             'n_estimators': [75, 100, 125]
             }
scoring = {'accuracy', 'precision', 'recall', 'f1'}
xgb_t0 = time.time()
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='f1')
xgb_cv.fit(X_train, y_train)
with open(path+'waze_rf_val_model.pickle', 'wb') as to_write:
    pickle.dump(rf_val, to_write)

print ("Modeling:")
xgb_time = time.time()-xgb_t0
print ("time_elapse:",xgb_time)
print ("The Best Parameters_XGB:")
print (xgb_cv)
# evaluate the reults
xgb_cv_results = make_results('XGBoost CV', xgb_cv)
# results = pd.read_csv(path+'Results.csv')

results = pd.concat([xgb_cv_results, rf_val_results])
results.sort_values(by=['F1'], ascending=False)
print ("Random Forest & GBM Comparison:")
print(results)
results.to_csv("waze_Results.csv")

# Predict on test data
print("xgb_cv_results:\n")

xgb_cv_preds = xgb_cv.predict(X_test)
print('F1 score final XGB model: ', f1_score(y_test, xgb_cv_preds))
print('Recall score final XGB model: ', recall_score(y_test, xgb_cv_preds))
print('Precision score final XGB model: ', precision_score(y_test, xgb_cv_preds))
print('Accuracy score final XGB model: ', accuracy_score(y_test, xgb_cv_preds))
##########################################################################################
# Get feature importances Random forest
feat_impt = rf_val.best_estimator_.feature_importances_

# Get indices of top 10 features
ind = np.argpartition(rf_val.best_estimator_.feature_importances_, -10)[-10:]

# Get column labels of top 10 features
feat = X.columns[ind]

# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]

y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)

y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")

ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")

plt.show()

#####


# feature importance
importances = xgb_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
ax.set_title("GBM")
fig.tight_layout()
plt.show()