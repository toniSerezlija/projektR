import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

metals = pd.read_table('C:/PROJEKT_R/projektR/measurements.txt')
feature_names = ['x', 'y', 'z']

""""
Q1 = metals['x'].quantile(0.25)
Q3 = metals['x'].quantile(0.75)
IQR = Q3-Q1
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
metals = metals[metals['x'] < Upper_Whisker]

Q1 = metals['y'].quantile(0.25)
Q3 = metals['y'].quantile(0.75)
IQR = Q3-Q1
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
metals = metals[metals['y'] < Upper_Whisker]

Q1 = metals['z'].quantile(0.25)
Q3 = metals['z'].quantile(0.75)
IQR = Q3-Q1
Lower_Whisker = Q1-1.5*IQR
Upper_Whisker = Q3+1.5*IQR
metals = metals[metals['z'] < Upper_Whisker]
"""

metals = metals.drop_duplicates(subset=feature_names)

X = metals[feature_names]
y = metals['name']

#
# Create training and test split
#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y)
#
# Standardize the data set
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#
# Fit the DTC model
#
clf = DecisionTreeClassifier().fit(X_train, y_train)
#
# Get the predictions
#
y_pred = clf.predict(X_test)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
#
# Print the confusion matrix
#
""""
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j],
                va='center', ha='center', size='xx-large')
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
"""

print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
      .format(clf.score(X_test, y_test)))
print('Precision: %.3f' % precision_score(y_test, y_pred, average='weighted'))
print('Recall: %.3f' % recall_score(y_test, y_pred, average='weighted'))
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average='weighted'))
