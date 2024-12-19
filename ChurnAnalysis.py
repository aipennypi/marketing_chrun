import numpy as np
import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
import pickle
path = '/'
import matplotlib.pyplot as plt
# from sklearn.tree import plot_tree

import seaborn as sns
# from sklearn.model_selection import GridSearchCV, train_test_split

# from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.preprocessing import StandardScaler
import time

import tensorflow as tf

pd.set_option('display.max_columns', None)


# Read the data
df=pd.read_csv("waze_dataset.csv")

print (df.head())
print ("info:\n",df.info())
print ("more information:\n",df.describe())
print ("data type:\n",df.dtypes)
print ("Column:\n",df.columns)
print ("row * column:\n",df.shape)

df=df.dropna()
# feature transformation
df['label']=df['label'].replace({'retained': 0, 'churned': 1})
df['device']=df['device'].replace({'Android': 0, 'iPhone': 1})

# build model

print(tf.version.VERSION)
model = tf.keras.Sequential([
              tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax')
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

print (model.summary())
history = model.fit(train_dataset, validation_data=eval_dataset, epochs=10)

def training_plot(metrics, history):
    f, ax = plt.subplots(1, len(metrics), figsize=(15,5))
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric])
        ax[idx].set_xlabel("Epochs")
        # ax[idx].set_ylabel(metric, fontweight='bold', fontsize=20)
        ax[idx].plot(history.history['val_' + metric], ls='dashed');
        ax[idx].legend([metric, 'val_' + metric], fontsize=20)

    plt.show()

training_plot(['loss', 'accuracy'], history)
