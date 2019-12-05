# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer

scoring = {'Accuracy': make_scorer(accuracy_score), 'Precision': make_scorer(precision_score, average='macro'),
           'Recall': make_scorer(recall_score, average='macro'), 'F1': make_scorer(f1_score, average='macro')}
random = 73
#
plt.rcParams["patch.force_edgecolor"] = True

data = pd.read_csv('iris/dataset/iris.data', names=['sep_length', 'sep_width', 'pet_length', 'pet_width', 'type'])
numeric = ['sep_length', 'sep_width', 'pet_length', 'pet_width']
target = 'type'

# %% Heatmap of Correlation
data.describe()
correlation_matrix = data[numeric].corr().round(2)
sns.heatmap(correlation_matrix, linewidths=1, annot=True)
# fix for cut off boxes
b, t = plt.ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
plt.ylim(b, t)  # update the ylim(bottom, top) values
plt.show()

# %% Visualizing Data
# Sepal
fig = data[data['type'] == 'Iris-setosa'].plot(kind='scatter', x='sep_length', y='sep_width', color='red',
                                               label='Setosa')
data[data['type'] == 'Iris-versicolor'].plot(kind='scatter', x='sep_length', y='sep_width', color='blue',
                                             label='Versicolor', ax=fig)
data[data['type'] == 'Iris-virginica'].plot(kind='scatter', x='sep_length', y='sep_width', color='green',
                                            label='Virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal: Length vs. Width")
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

# Petal
fig = data[data['type'] == 'Iris-setosa'].plot(kind='scatter', x='pet_length', y='pet_width', color='red',
                                               label='Setosa')
data[data['type'] == 'Iris-versicolor'].plot(kind='scatter', x='pet_length', y='pet_width', color='blue',
                                             label='Versicolor', ax=fig)
data[data['type'] == 'Iris-virginica'].plot(kind='scatter', x='pet_length', y='pet_width', color='green',
                                            label='Virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal: Length vs. Width")
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

# %% KNN CV NP, Comparing different metrics
k = np.arange(1, 41)
metric = ['euclidean', 'chebyshev', 'manhattan']

knn = GridSearchCV(KNeighborsClassifier(), dict(n_neighbors=k, metric=metric),
                   cv=10, n_jobs=-1)
knn.fit(data[numeric], data[target])
print('Best Mean Score Without Preprocessing', knn.best_score_, 'Model', knn.best_estimator_)
best_estimator = knn.best_estimator_

knn_results = pd.DataFrame(knn.cv_results_)
sns.lineplot('param_n_neighbors', 'mean_test_score', 'param_metric', style='param_metric', data=knn_results)
plt.show()

# %% KNN CV P, Comparing euclidean metrics P and NP
sns.lineplot('param_n_neighbors', 'mean_test_score', data=knn_results[knn_results['param_metric'] == 'euclidean'])

classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(), KNeighborsClassifier())
knn = GridSearchCV(classifier_pipeline, dict(kneighborsclassifier__n_neighbors=k, kneighborsclassifier__metric=metric),
                   cv=10, n_jobs=-1)

knn.fit(data[numeric], data[target])

print('Best Mean Score With Preprocessing', knn.best_score_, 'Model', knn.best_estimator_)
knn_results = pd.DataFrame(knn.cv_results_)

sns.lineplot('param_kneighborsclassifier__n_neighbors', 'mean_test_score', data=knn_results[knn_results['param_kneighborsclassifier__metric'] == 'euclidean'])
plt.legend(['Without Preprocessing', 'With Preprocessing'])
plt.show()

# %% KNN Scorer and Time
results = cross_validate(best_estimator, data[numeric], data[target], scoring=scoring, cv=10)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% KNN HO
X_train, X_test, y_train, y_test = train_test_split(data[numeric], data[target], test_size=0.2, random_state=random,
                                                    stratify=data[target])
best_estimator.fit(X_train, y_train)
print('Best Score Hold Out', best_estimator.score(X_test, y_test))

# %% RF CV NP, Comparing different metrics
k = np.arange(1, 41)
metric = ['euclidean', 'chebyshev', 'manhattan']

knn = GridSearchCV(KNeighborsClassifier(), dict(n_neighbors=k, metric=metric),
                   cv=10, n_jobs=-1)
knn.fit(data[numeric], data[target])
print('Best Mean Score Without Preprocessing', knn.best_score_, 'Model', knn.best_estimator_)
best_estimator = knn.best_estimator_

knn_results = pd.DataFrame(knn.cv_results_)
sns.lineplot('param_n_neighbors', 'mean_test_score', 'param_metric', style='param_metric', data=knn_results)
plt.show()

# %% KNN CV P, Comparing euclidean metrics P and NP
sns.lineplot('param_n_neighbors', 'mean_test_score', data=knn_results[knn_results['param_metric'] == 'euclidean'])

classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(), KNeighborsClassifier())
knn = GridSearchCV(classifier_pipeline, dict(kneighborsclassifier__n_neighbors=k, kneighborsclassifier__metric=metric),
                   cv=10, n_jobs=-1)

knn.fit(data[numeric], data[target])

print('Best Mean Score With Preprocessing', knn.best_score_, 'Model', knn.best_estimator_)
knn_results = pd.DataFrame(knn.cv_results_)

sns.lineplot('param_kneighborsclassifier__n_neighbors', 'mean_test_score', data=knn_results[knn_results['param_kneighborsclassifier__metric'] == 'euclidean'])
plt.legend(['Without Preprocessing', 'With Preprocessing'])
plt.show()

# %% KNN Scorer and Time
results = cross_validate(best_estimator, data[numeric], data[target], scoring=scoring, cv=10)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% KNN HO
X_train, X_test, y_train, y_test = train_test_split(data[numeric], data[target], test_size=0.2, random_state=random,
                                                    stratify=data[target])
best_estimator.fit(X_train, y_train)
print('Best Score Hold Out', best_estimator.score(X_test, y_test))



# scores = cross_val_score(DecisionTreeClassifier(), data[numeric], data[target], cv=10).mean()

# scores = cross_val_score(MLPClassifier(max_iter=1000), data[numeric], data[target], cv=10).mean()
