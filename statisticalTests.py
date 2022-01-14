from statistics import LinearRegression
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from scipy.stats import shapiro

dataframe = pd.read_table(
    'C:/PROJEKT_R/projektR/newDataset.txt', sep="\s+")
data = dataframe.values

print(shapiro(data))

X = data[:, :-1]
y = data[:, -1]

#
# filter methods
#
constant_filter = VarianceThreshold(threshold=0.005)
data_constant = constant_filter.fit_transform(X)
print(data_constant.shape)
X_new1 = SelectPercentile(
    mutual_info_classif, percentile=10).fit_transform(data_constant, y)
print(X_new1.shape)
scalerS = StandardScaler()


X_train, X_test, y_train, y_test = train_test_split(
    data_constant, y, test_size=0.2)
X_train = scalerS.fit_transform(X_train)
X_test = scalerS.transform(X_test)

clf = RandomForestClassifier(
    n_estimators=100, max_depth=5).fit(X_train, y_train)
dtc = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
gnb = GaussianNB().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
model = LinearRegression()
#
# embedded methods
#
skf = StratifiedKFold(n_splits=10)
lasso = LassoCV(cv=skf, random_state=42).fit(X, y)
lr = LogisticRegression(C=10, class_weight='balanced',
                        max_iter=10000, random_state=42)
preds = cross_val_predict(gnb, X[:, np.where(lasso.coef_ != 0)[0]], y, cv=skf)
print(classification_report(y, preds))

#
# Get the predictions
#
y_pred = clf.predict(X_test)
y_pred1 = dtc.predict(X_test)
y_pred2 = gnb.predict(X_test)
y_pred3 = svm.predict(X_test)
#
# Calculate the accuracy
#
acc1 = accuracy_score(y_test, y_pred)
acc2 = accuracy_score(y_test, y_pred1)
#
# Calculate the confusion matrix
#
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred2)
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

print("Paired t-test Resampled")
t, p = paired_ttest_5x2cv(
    estimator1=gnb, estimator2=svm, X=X, y=y, random_seed=42)
print(f"t statistic: {t}, p-value: {p}\n")
