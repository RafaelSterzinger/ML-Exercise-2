# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer
from scipy.stats import uniform, truncnorm, randint

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

sns.lineplot('param_kneighborsclassifier__n_neighbors', 'mean_test_score',
             data=knn_results[knn_results['param_kneighborsclassifier__metric'] == 'euclidean'])
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

# %% RF CV approx params
param_grid = {
    # randomly sample numbers from 4 to 204 estimators
    'n_estimators': randint(100, 130),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'min_samples_split': uniform(0.01, 0.199)
}

rf = RandomizedSearchCV(RandomForestClassifier(random_state=random, criterion='gini'), param_grid, cv=10,
                        n_jobs=-1, random_state=random, n_iter=100)

rf.fit(data[numeric], data[target])
print('Best Mean Score Without Preprocessing', rf.best_score_, 'Model', rf.best_estimator_)
# %% RF CV NP
param_grid = {
    'n_estimators': np.arange(100, 150),
    'criterion': ['gini', 'entropy']
}

rf = GridSearchCV(RandomForestClassifier(random_state=random, min_samples_split=0.01, max_features=0.33), param_grid,
                  cv=5,
                  n_jobs=-1)
rf.fit(data[numeric], data[target])
print('Best Mean Score Without Preprocessing', rf.best_score_, 'Model', rf.best_estimator_)
best_estimator = rf.best_estimator_

rf_results = pd.DataFrame(rf.cv_results_)
sns.lineplot('param_n_estimators', 'mean_test_score', 'param_criterion', style='param_criterion', data=rf_results)
plt.show()

# %% RF CV NP
param_grid = {
    'n_estimators': np.arange(50, 150),
}

rf = GridSearchCV(RandomForestClassifier(random_state=random, max_features=0.5, min_samples_split=0.01), param_grid,
                  cv=5,
                  n_jobs=-1)
rf.fit(data[numeric], data[target])
print('Best Mean Score Without Preprocessing', rf.best_score_, 'Model', rf.best_estimator_)

rf_results = pd.DataFrame(rf.cv_results_)
sns.lineplot('param_n_estimators', 'mean_test_score', data=rf_results)
plt.show()

# %% RF CV P
sns.lineplot('param_n_estimators', 'mean_test_score', data=rf_results[rf_results['param_criterion'] == 'gini'])

classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(),
                                    RandomForestClassifier(random_state=random, min_samples_split=0.01,
                                                           max_features=0.33))
param_grid = {
    'randomforestclassifier__n_estimators': np.arange(100, 150),
}

rf = GridSearchCV(classifier_pipeline, param_grid,
                  cv=5,
                  n_jobs=-1)
rf.fit(data[numeric], data[target])

print('Best Mean Score Without Preprocessing', rf.best_score_, 'Model', rf.best_estimator_)

rf_results = pd.DataFrame(rf.cv_results_)
sns.lineplot('param_randomforestclassifier__n_estimators', 'mean_test_score', data=rf_results)
plt.legend(['Without Preprocessing', 'With Preprocessing'])
plt.show()

# %% RF Scorer and Time
results = cross_validate(best_estimator, data[numeric], data[target], scoring=scoring, cv=10)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% RF HO
X_train, X_test, y_train, y_test = train_test_split(data[numeric], data[target], test_size=0.2, random_state=random,
                                                    stratify=data[target])
best_estimator.fit(X_train, y_train)
print('Best Score Hold Out', best_estimator.score(X_test, y_test))

# %% MLP CV approx params
param_grid = {
    'hidden_layer_sizes': [(3, 4, 3), (4, 4, 4), (4, 3, 4)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.01, 0.001, 0.0001]
}

mlp = RandomizedSearchCV(MLPClassifier(max_iter=5000, random_state=random), param_grid, cv=3,
                         n_jobs=-1, random_state=random)

mlp.fit(data[numeric], data[target])
print('Best Mean Score Without Preprocessing', mlp.best_score_, 'Model', mlp.best_estimator_)

# %% MLP CV NP
param_grid = {
    'hidden_layer_sizes': [(3, 4, 3), (4, 4, 4), (4, 3, 4)],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
}

mlp = GridSearchCV(
    MLPClassifier(alpha=0.001, solver='sgd', learning_rate='constant', max_iter=4000, random_state=random), param_grid,
    cv=3,
    n_jobs=-1)

mlp.fit(data[numeric], data[target])
best_estimator = mlp.best_estimator_

print('Best Mean Score Without Preprocessing', mlp.best_score_, 'Model', mlp.best_estimator_)
mlp_results = pd.DataFrame(mlp.cv_results_)

sns.barplot('param_hidden_layer_sizes', 'mean_test_score', 'param_activation', data=mlp_results)
plt.show()

# %% MLP CV P
classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(),
                                    MLPClassifier(alpha=0.001, solver='sgd', learning_rate='constant', max_iter=4000,
                                                  random_state=random))

param_grid = {
    'mlpclassifier__hidden_layer_sizes': [(3, 4, 3), (4, 4, 4), (4, 3, 4)],
    'mlpclassifier__activation': ['tanh', 'relu', 'logistic', 'identity'],
}

mlp1 = GridSearchCV(classifier_pipeline, param_grid, cv=3,
                    n_jobs=-1)

mlp1.fit(data[numeric], data[target])

print('Best Mean Score With Preprocessing', mlp1.best_score_, 'Model', mlp1.best_estimator_)
mlp1_results = pd.DataFrame(mlp1.cv_results_)

plotdata = mlp_results[mlp_results['param_hidden_layer_sizes'] == (4, 3, 4)]
temp = mlp1_results[mlp1_results['param_mlpclassifier__hidden_layer_sizes'] == (3, 4, 3)]
temp = temp.rename(columns={'param_mlpclassifier__hidden_layer_sizes': 'param_hidden_layer_sizes',
                            'param_mlpclassifier__activation': 'param_activation'})
plotdata = plotdata.append(temp)
plotdata['param_hidden_layer_sizes'] = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']

sns.barplot('param_activation', 'mean_test_score', 'param_hidden_layer_sizes', data=plotdata)
plt.legend(['Without Preprocessing (4,3,4)', 'With Preprocessing (3,4,3)'])
plt.show()
# %%
results = cross_validate(best_estimator, data[numeric], data[target], scoring=scoring, cv=10)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% KNN HO
X_train, X_test, y_train, y_test = train_test_split(data[numeric], data[target], test_size=0.2, random_state=random,
                                                    stratify=data[target])
best_estimator.fit(X_train, y_train)
print('Best Score Hold Out', best_estimator.score(X_test, y_test))
