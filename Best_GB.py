import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

# Load dataset
df_new = pd.read_csv("data_set.csv")
duplicates = df_new.duplicated().any()
pd.set_option('display.max_rows', 50)
df = df_new.copy()

# Remove id column
df = df.drop('id', axis=1)

# Remove outliers for BMI data
mean_bmi = round(df['bmi'].mean(), 1)
df_new['bmi'].fillna(mean_bmi, inplace=True)
df = df[(df['bmi'] >= 15) & (df['bmi'] <= 70)]
najwieksze_bmi = df['bmi'].max()
najmniejsze_bmi = df['bmi'].min()

# Numerical representation
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['ever_married'] = label_encoder.fit_transform(df['ever_married'])
df['work_type'] = label_encoder.fit_transform(df['work_type'])
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type'])
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status'])

# Normalization into 0-1 range
scaler = MinMaxScaler()
df['age'] = scaler.fit_transform(df[['age']])
df['avg_glucose_level'] = scaler.fit_transform(df[['avg_glucose_level']])
df['bmi'] = scaler.fit_transform(df[['bmi']])

# Class balance
class_0 = df[df['stroke'] == 0]
class_1 = df[df['stroke'] == 1]
class_1_oversampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
oversampled_data = pd.concat([class_0, class_1_oversampled])
df = oversampled_data
sns.countplot(x='stroke', data=df).set(title='Countplot')
value_counts = df['stroke'].value_counts()
print(df.shape)

# Data dimensionality reduction
X = df.drop(['stroke', 'gender', 'hypertension', 'Residence_type', 'smoking_status'], axis=1)
target = df['stroke']

# Division of data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.15, random_state=42)

# Classification
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Documenting the results
score = clf.score(X_test, y_test)
cm = metrics.confusion_matrix(y_test, clf.predict(X_test))
cross_val_scores = cross_val_score(clf, X_test, y_test, cv=5)

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
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Gradient Boosting')
display.plot(ax=ax)
ax.legend(fontsize=20)
plt.xlabel("specificity")
plt.ylabel("sensitivity")
plt.title("ROC curve")
plt.show()
pole = roc_auc_score(y_test, y_pred)
print('Area under ROC curve:', round(pole, 3))