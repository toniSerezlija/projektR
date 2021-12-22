#from scipy.sparse import data
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC

dataframe = pd.read_table(
    'C:/PROJEKT_R/projektR/binaryClassificationWithMultipliedAxes.txt', sep="\s+")


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


dataframe = clean_dataset(dataframe)

data = dataframe.values

X = data[:, :-1]
y = data[:, -1]

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
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
#
# Fit the DTC model
#
clf = RandomForestClassifier(
    n_estimators=100, max_depth=5).fit(X_train, y_train)
dtc = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
gnb = GaussianNB().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
#
# Get the predictions
#
y_pred = clf.predict(X_test)
y_pred1 = dtc.predict(X_test)
y_pred2 = gnb.predict(X_test)
y_pred3 = svm.predict(X_test)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred3)
#
# Print the confusion matrix
#

group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in
                conf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in
                     conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
ax = sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix for binary classification\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
# Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])
# Display the visualization of the Confusion Matrix.
plt.show()


print('Accuracy of Random Forrest classifier on training set : {:.2f}'
      .format(clf.score(X_train, y_train)))
print('Accuracy of Random Forrest classifier on test set : {:.2f}'
      .format(clf.score(X_test, y_test)))
print("\n")
print('Accuracy of Decision Tree classifier on training set : {:.2f}'
      .format(dtc.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set : {:.2f}'
      .format(dtc.score(X_test, y_test)))
print("\n")
print('Precision for Random Forrest classifier : %.3f' %
      precision_score(y_test, y_pred, average='weighted'))
print('Recall for Random Forrest classifier : %.3f' %
      recall_score(y_test, y_pred, average='weighted'))
print('F1 Score for Random Forrest classifier : %.3f' %
      f1_score(y_test, y_pred, average='weighted'))
print("\n")
print('Precision for Decision Tree classifier : %.3f' %
      precision_score(y_test, y_pred1, average='weighted'))
print('Recall for Decision Tree classifier : %.3f' %
      recall_score(y_test, y_pred1, average='weighted'))
print('F1 Score for Decision Tree classifier : %.3f' %
      f1_score(y_test, y_pred1, average='weighted'))
print("\n")
