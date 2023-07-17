import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

#Load Data
df_new = pd.read_csv("data_set.csv")
duplicates = df_new.duplicated().any()
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_rows', 50)
df = df_new.copy()

#Remove id column
df = df.drop('id', axis=1)
mean_bmi = round(df['bmi'].mean(), 1)
df_new['bmi'].fillna(mean_bmi, inplace=True)

#Remove outliers for BMI data
df = df[(df['bmi'] >= 15) & (df['bmi'] <= 70)]
najwieksze_bmi = df['bmi'].max()
najmniejsze_bmi = df['bmi'].min()

#Numerical representation
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
df['work_type'] = label_encoder.fit_transform(df['work_type'])
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])

#Normalization into 0-1 range
scaler = MinMaxScaler()
df['age'] = scaler.fit_transform(df[['age']])
df['avg_glucose_level'] = scaler.fit_transform(df[['avg_glucose_level']])
df['bmi'] = scaler.fit_transform(df[['bmi']])

#Class balance
class_0 = df[df['stroke'] == 0]
class_1 = df[df['stroke'] == 1]
class_0_undersampled = resample(class_0, replace=False, n_samples=len(class_1), random_state=42)
undersampled_data = pd.concat([class_0_undersampled, class_1])
df = undersampled_data
sns.countplot(x='stroke', data=df).set(title='Countplot')
value_counts = df['stroke'].value_counts()
print("Data shape:", df.shape)

#Data dimensionality reduction
X = df.drop(['stroke', 'gender', 'hypertension', 'Residence_type', 'smoking_status'], axis=1)
target = df['stroke']

#Division of data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.15, random_state=42)

#Grid-search for best parameters
param_grid = {
    "kernel": ['linear', 'rbf', 'poly'],
    "C": [0.1, 1, 10],
    "gamma": [0.1, 0.01, 0.001],
    "degree": [3, 4, 5]
}

clf = svm.SVC()
grid_search = GridSearchCV(clf, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

#Classification
clf_best = svm.SVC(**best_params)
clf_best.fit(X_train, y_train)
y_pred = clf_best.predict(X_test)

#Documenting the results
print("Accuracy metric:", round(metrics.accuracy_score(y_test, y_pred), 3))
print("Precision metric:", round(metrics.precision_score(y_test, y_pred), 3))
print("Recall metric:", round(metrics.recall_score(y_test, y_pred), 3))
print("F1 metric:", round(f1_score(y_test, y_pred, average='macro'), 3))

score = clf_best.score(X_test, y_test)
cm = metrics.confusion_matrix(y_test, clf_best.predict(X_test))
cross_val_scores = cross_val_score(clf_best, X_test, y_test, cv=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(cm)
ax1.grid(False)
ax1.set_title('Confusion matrix', fontsize=16)
ax1.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax1.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax1.set_ylim(1.5, -0.5)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm[i, j], ha='center', va='center', color='red')

ax2.bar(range(len(cross_val_scores)), cross_val_scores)
ax2.xaxis.set(label='Cross validation batch')
ax2.yaxis.set(label='Score')
ax2.set_title('Cross-Validation Accuracy Scores', fontsize=16)
ax2.set_ylim(0.0, 1.0)
ax2.tick_params(axis='x', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()
print("\nClassification report: ")
print(metrics.classification_report(y_test, y_pred))
print("\nScore: ")
print(round(score, 3), '\n')

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(12, 8))
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='SVM')
display.plot(ax=ax)
ax.legend(fontsize=20)
plt.xlabel("specificity")
plt.ylabel("sensitivity")
plt.title("ROC, curve")
plt.show()
pole = roc_auc_score(y_test, y_pred)
print('Area under ROC curve:', round(pole, 3))