# import packages
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# load data
import os
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv('/content/drive/MyDrive/Dataset/diabetes.csv')

x = data.drop(['Outcome'], axis=1)
y = data.Outcome
x.shape
x.head()
y.head()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

from math import log2
from sklearn.ensemble import RandomForestClassifier

# 以 BayesSearchCV 找出 Best Hyperparameters
params = {
    'ccp_alpha': 0.0,
    'max_depth': 10,
    'max_leaf_nodes': 100,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 0.1,
    'min_samples_split': 8,
    'min_weight_fraction_leaf': 0.1,
    'n_estimators': 50,
    'verbose': 2
}

rf = RandomForestClassifier(**params)
rf.fit(x_train, y_train)

y_pred =rf.predict(x_test)
y_pred

print('Accuracy',np.mean(y_pred == y_test)*100)

target_name = ['non-Diabetes', 'Diabetes']
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=target_name))
