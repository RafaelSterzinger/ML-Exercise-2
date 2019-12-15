# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer,accuracy_score,precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from scipy.stats import uniform, truncnorm, randint

from sklearn.pipeline import Pipeline

random = 47

scoring = {'Accuracy': make_scorer(accuracy_score), 'Precision': make_scorer(precision_score, average='macro'),
           'Recall': make_scorer(recall_score, average='macro'), 'F1': make_scorer(f1_score, average='macro')}

train = pd.read_csv('dataset/breast-cancer-diagnostic.shuf.lrn.csv')
train.columns = train.columns.str.strip()

# %% Pre-Processing
y = train['class']
X = train.drop(['ID', 'class'], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X)
X_min_max = min_max_scaler.transform(X)

# %% Count-Plot of Target
sns.countplot(y, label="Count")  # M = 212, B = 357
plt.savefig("plots/countplot.png")
plt.show()

# %% violinplot minmax transformed
# first ten features
data_dia = y
data = (X - X.min()) / (X.max() - X.min())
data = pd.concat([y, data.iloc[:, 0:10]], axis=1)
data = pd.melt(data,
               id_vars="class",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.violinplot(x="features", y="value", hue="class", data=data, split=True, inner="quart")
plt.xticks(rotation=40)
plt.savefig("plots/violinplot.png")
plt.show()

# %% violineplot normal transformed
data_dia = y
data = (X - X.mean()) / (X.std())
data = pd.concat([y, data.iloc[:, 0:10]], axis=1)
data = pd.melt(data,
               id_vars="class",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10, 10))
sns.violinplot(x="features", y="value", hue="class", data=data, split=True, inner="quart")
plt.xticks(rotation=90)
plt.savefig("plots/violineplot_normaltransformed.png")
plt.show()

# %% Boxplot
plt.figure(figsize=(10, 10))
sns.boxplot(x="features", y="value", hue="class", data=data)
plt.xticks(rotation=90)
plt.savefig("plots/boxplot.png")
plt.show()

# %%
sns.jointplot(X['concavityWorst'], X['concavePointsWorst'], kind="regg", color="#ce1414")
plt.savefig("plots/corr_comparision.png")
plt.show()

# %%
plt.figure(figsize=(10, 10))
sns.swarmplot(x="features", y="value", hue="class", data=data)
plt.xticks(rotation=90)
plt.savefig("plots/swarmplot.png")
plt.show()

# %%
drop_list1 = ['perimeterMean', 'radiusMean', 'compactnessMean', 'concavePointsMean', 'radiusStdErr', 'perimeterStdErr',
              'radiusWorst', 'perimeterWorst', 'compactnessWorst', 'concavePointsWorst', 'compactnessStdErr',
              'concavePointsStdErr', 'textureWorst', 'areaWorst']
X_best = X.drop(drop_list1, axis=1)  # do not modify x, we will use it later

#%% Comparing Feature Selection
k = np.arange(1, 40)
metric = ['euclidean', 'chebyshev', 'manhattan']

knn = GridSearchCV(KNeighborsClassifier(), dict(n_neighbors=k, metric=metric),
                   cv=10, n_jobs=-1)
knn.fit(X, y)
print('Best Mean Score All Features', knn.best_score_, 'Model', knn.best_estimator_)

knn_results = pd.DataFrame(knn.cv_results_)
sns.lineplot('param_n_neighbors', 'mean_test_score', data=knn_results[knn_results['param_metric'] == 'manhattan'])

knn = GridSearchCV(KNeighborsClassifier(), dict(n_neighbors=k, metric=metric),
                   cv=10, n_jobs=-1)
knn.fit(X_best, y)
print('Best Mean Score With Selected Features', knn.best_score_, 'Model', knn.best_estimator_)

knn_results = pd.DataFrame(knn.cv_results_)
sns.lineplot('param_n_neighbors', 'mean_test_score',
             data=knn_results[knn_results['param_metric'] == 'manhattan'])
plt.legend(['With all Features', 'With selected Features'])
plt.savefig("plots/knn_feature_comparision.png")
plt.show()

# %% Random Forest feature selection
rf = RandomForestClassifier(n_estimators=100)
rf_ecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy', n_jobs=-1)
rf_ecv = rf_ecv.fit(X, y)

print('Optimal number of features :', rf_ecv.n_features_)
print('Best features :', X.columns[rf_ecv.support_])

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rf_ecv.grid_scores_) + 1), rf_ecv.grid_scores_)
plt.savefig("plots/rf_feature_selection.png")
plt.show()

#%% KNN NP
k = np.arange(1, 30)
metric = ['euclidean', 'chebyshev', 'manhattan']

knn = GridSearchCV(KNeighborsClassifier(), dict(n_neighbors=k, metric=metric),
                   cv=10, n_jobs=-1)
knn.fit(X, y)
print('Best Mean Score Without Preprocessing', knn.best_score_, 'Model', knn.best_estimator_)

# %% KNN P
pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
                     ('c', KNeighborsClassifier(weights='distance'))])
k = range(1, 30)
metric = ['euclidean', 'chebyshev', 'manhattan']

grid_search_dict = dict(c__n_neighbors=k, c__metric=metric)

knn = GridSearchCV(pipeline, grid_search_dict, cv=10, n_jobs=-1)
knn.fit(X, y)
best_estimator = knn.best_estimator_
print('Best Mean Score with Preprocessing', knn.best_score_, 'Model',
      best_estimator)

knn_results = pd.DataFrame(knn.cv_results_)
sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric', style='param_c__metric', data=knn_results)
plt.savefig("plots/knn_p_comparision.png")
plt.show()

# %% KNN Scorer and Time
results = cross_validate(best_estimator, X_min_max, y, scoring=scoring, cv=10)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% KNN HO
X_train, X_test, y_train, y_test = train_test_split(X_min_max, y, test_size=0.2, random_state=random,
                                                    stratify=y)
best_estimator.fit(X_train, y_train)
print('Best Score Hold Out', best_estimator.score(X_test, y_test))

# %% RF CV approx params
param_grid = {
    # randomly sample numbers from 4 to 204 estimators
    'n_estimators': randint(1, 50),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    'min_samples_split': uniform(0.01, 0.199)
}

rf = RandomizedSearchCV(RandomForestClassifier(random_state=random, criterion='gini'), param_grid, cv=4,
                        n_jobs=-1, random_state=random, n_iter=100,verbose=True)
rf.fit(X, y)

print('Best Mean Score Approx Params', rf.best_score_, 'Model', rf.best_estimator_)

# %% RF CV NP
param_grid = {
    'n_estimators': np.arange(1,50),
    'max_features': [0.1,0.2,0.3,0.4,0.5],
}

rf = GridSearchCV(RandomForestClassifier(random_state=random, min_samples_split=0.04), param_grid,
                  cv=5,
                  n_jobs=-1,verbose=True)
rf.fit(X, y)
print('Best Mean Score Without Preprocessing', rf.best_score_, 'Model', rf.best_estimator_)
best_estimator = rf.best_estimator_

rf_results = pd.DataFrame(rf.cv_results_)
rf_results['param_max_features'] = list(map(lambda x: str(x*100) + ' %',rf_results['param_max_features']))
sns.lineplot('param_n_estimators', 'mean_test_score','param_max_features',style='param_max_features', data=rf_results)
plt.savefig("plots/rf_np_comparision.png")
plt.show()

# %% RF CV P
classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(),
                                    RandomForestClassifier(random_state=random, min_samples_split=0.04))

param_grid = {
    'randomforestclassifier__n_estimators': np.arange(1,50),
    'randomforestclassifier__max_features': [0.1,0.2,0.3,0.4,0.5],
}

rf = GridSearchCV(classifier_pipeline, param_grid,
                  cv=5,
                  n_jobs=-1,verbose=True)
rf.fit(X, y)
print('Best Mean Score With Preprocessing', rf.best_score_, 'Model', rf.best_estimator_)

# %% RF Scorer and Time
results = cross_validate(best_estimator, X, y, scoring=scoring, cv=10)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% RF HO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random,
                                                    stratify=y)
best_estimator.fit(X_train, y_train)
print('Best Score Hold Out', best_estimator.score(X_test, y_test))

# %% MLP CV approx params
param_grid = {
    'hidden_layer_sizes': [(15,15,15),(30, 15, 30), (15, 30, 15), (30, 30, 30)],
    'activation': ['tanh', 'relu','logistic','identity'],
    'solver': ['sgd', 'adam'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.01, 0.001, 0.0001]
}

mlp = RandomizedSearchCV(MLPClassifier(max_iter=4000, random_state=random), param_grid, cv=5,
                         n_jobs=-1, random_state=random,verbose=True)

mlp.fit(X, y)
print('Best Mean Score Approx Params', mlp.best_score_, 'Model', mlp.best_estimator_)

# %% MLP CV NP
param_grid = {
    'hidden_layer_sizes': [(15,15,15),(30, 15, 30), (15, 30, 15), (30, 30, 30)],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
}

mlp = GridSearchCV(
    MLPClassifier(alpha=0.001, solver='adam', learning_rate='constant', max_iter=3000, random_state=random), param_grid,
    cv=3,
    n_jobs=-1, verbose=True)

mlp.fit(X, y)

print('Best Mean Score Without Preprocessing', mlp.best_score_, 'Model', mlp.best_estimator_)
mlp_results = pd.DataFrame(mlp.cv_results_)

# %% MLP CV P
classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(),
                                    MLPClassifier(alpha=0.001, solver='adam', learning_rate='constant', max_iter=3000,
                                                  random_state=random))

param_grid = {
    'mlpclassifier__hidden_layer_sizes': [(15,15,15),(30, 15, 30), (15, 30, 15), (30, 30, 30)],
    'mlpclassifier__activation': ['tanh', 'relu', 'logistic', 'identity'],
}

mlp1 = GridSearchCV(classifier_pipeline, param_grid, cv=3,
                    n_jobs=-1, verbose=True)

mlp1.fit(X,y)
best_estimator = mlp1.best_estimator_

print('Best Mean Score With Preprocessing', mlp1.best_score_, 'Model', mlp1.best_estimator_)
mlp1_results = pd.DataFrame(mlp1.cv_results_)

sns.barplot('param_mlpclassifier__hidden_layer_sizes', 'mean_test_score', 'param_mlpclassifier__activation', data=mlp1_results)
plt.savefig("plots/mlp_p_comparision.png")
plt.show()

plotdata = mlp_results[mlp_results['param_hidden_layer_sizes'] == (15, 15, 15)]
temp = mlp1_results[mlp1_results['param_mlpclassifier__hidden_layer_sizes'] == (30, 15, 30)]
temp = temp.rename(columns={'param_mlpclassifier__hidden_layer_sizes': 'param_hidden_layer_sizes',
                            'param_mlpclassifier__activation': 'param_activation'})
plotdata = plotdata.append(temp)
plotdata['param_hidden_layer_sizes'] = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']

sns.barplot('param_activation', 'mean_test_score', 'param_hidden_layer_sizes', data=plotdata)
plt.legend(['Without Preprocessing (15,15,15)', 'With Preprocessing (30,15,30)'])
plt.savefig("plots/mlp_np_p_comparision.png")
plt.show()

# %% MLP Scorer and Time
results = cross_validate(best_estimator, X, y, scoring=scoring, cv=10)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% MLP HO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random,
                                                    stratify=y)
best_estimator.fit(X_train, y_train)
print('Best Score Hold Out', best_estimator.score(X_test, y_test))

# %% Kaggle Score 0.97647
#X_scale = min_max_scaler.fit_transform(X)
#test = pd.read_csv('breast/dataset/breast-cancer-diagnostic.shuf.tes.csv').drop('ID', axis=1)
#test.columns = test.columns.str.strip()
#sol = pd.read_csv('breast/dataset/breast-cancer-diagnostic.shuf.sol.ex.csv')

#mlp = MLPClassifier(
#    hidden_layer_sizes=(30,15,30),
#    max_iter=3000,
#    alpha=0.001,
#    activation='relu',
#    solver='adam',
#)

#mlp.fit(X_scale, y)
#prediction = mlp.predict(min_max_scaler.transform(test))

#sol['class'] = prediction
#sol.to_csv("breast/dataset/sol.csv", index=False)
