from pprint import pprint

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets, ensemble, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


def summary(r, p):
    # matrix
    matrix = confusion_matrix(r, p)
    ax = sns.heatmap(matrix, annot=True, cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ax.xaxis.set_ticklabels(['Iris Setosa', 'Iris Versicolour', 'Iris Virginica'])
    ax.yaxis.set_ticklabels(['Iris Setosa', 'Iris Versicolour', 'Iris Virginica'])

    plt.show()


Iris_data = datasets.load_iris()

X = Iris_data.data
Y = Iris_data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=101)

# Without using GridSearchCV
print("Without...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

summary(y_test, y_pred)

# With GridSearchCV
print("With...")
model_parameters = {
    'n_estimators': [10, 20, 50, 100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 3, 4],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', 'balanced_subsample'],
    'ccp_alpha': [0.0, 0.1, 0.2],
    'n_jobs': [1, -1],
    'verbose': [0]
}

clf = RandomForestClassifier(bootstrap=True)
grid = GridSearchCV(clf, model_parameters, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print(metrics.f1_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred)))

print(classification_report(y_test, y_pred))

pprint(grid.best_estimator_.get_params())

summary(y_test, y_pred)
