import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint, truncnorm, uniform

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve, cross_validate

scoring = {'Accuracy': make_scorer(accuracy_score), 'Precision': make_scorer(precision_score, average='macro'),
           'Recall': make_scorer(recall_score, average='macro'), 'F1': make_scorer(f1_score, average='macro')}

random = 123
plt.rcParams["patch.force_edgecolor"] = True

# %% load datasets
dataset_path = "dataset/"
train = pd.read_csv(dataset_path + "amazon_review_ID.shuf.lrn.csv")

X = train.drop(['Class', 'ID'], axis=1)
y = train['Class']

# %% plot target
plt.figure(figsize=(10, 6))
chart = sns.countplot(y, label="Count")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('plots/target.png')
plt.show()

# %% Select K Best
selector = SelectKBest(chi2, k=1000)
X_best_k = selector.fit_transform(X, y)

# %%
scaler_min_max = MinMaxScaler()
scaler_min_max.fit(X, y)
X_transformed_min_max = scaler_min_max.transform(X)

# %% Random Forest feature selection
# The "accuracy" scoring is proportional to the number of correct classifications
step_size = 1000
clf_rf_4 = RandomForestClassifier(random_state=random)
rf_ecv = RFECV(estimator=clf_rf_4, step=step_size, min_features_to_select=10, cv=3, scoring='accuracy',
               verbose=True, n_jobs=-1)  # 5-fold cross-validation
rf_ecv = rf_ecv.fit(X, y)

print('Optimal number of features :', rf_ecv.n_features_)
print('Best features :', X.columns[rf_ecv.support_])
best_rfecv_features = X.columns[rf_ecv.support_]

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rf_ecv.grid_scores_) * step_size, step_size), rf_ecv.grid_scores_)
plt.savefig('plots/rf_feature_selection.png')
plt.show()


# %% fixed Heatplot of found attributes
def plot_heatmap(df):
    correlation_matrix = df.corr().abs()
    sns.heatmap(correlation_matrix, linewidths=.5).get_figure()
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()


# %%
k = range(1, 60)
metric = ['euclidean', 'chebyshev', 'manhattan']
grid_search_dict = dict(c__n_neighbors=k, c__metric=metric)

# %% knn cv np, k and metrics
pipeline = Pipeline([('c', KNeighborsClassifier(weights='distance'))])
knn_np = GridSearchCV(pipeline, grid_search_dict, cv=5, n_jobs=-1)
knn_np.fit(X, y)
print('Best Mean Score without Preprocessing', knn_np.best_score_, 'Model', knn_np.best_estimator_)
knn_results_np = pd.DataFrame(knn_np.cv_results_)

# %% knn cv p, k and metrics
pipeline = Pipeline([('s', preprocessing.MinMaxScaler()),
                     ('c', KNeighborsClassifier(weights='distance'))])

knn_p = GridSearchCV(pipeline, grid_search_dict, cv=5, n_jobs=-1)
knn_p.fit(X[best_rfecv_features], y)
print('Best Mean Score with Preprocessing', knn_p.best_score_, 'Model', knn_p.best_estimator_)
knn_results_p = pd.DataFrame(knn_p.cv_results_)

# %%
sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric', data=knn_results_p)
plt.savefig('plots/knn_metrics.png')
plt.show()

# %% plot results
sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric',
             data=knn_results_np[knn_results_np['param_c__metric'] == 'manhattan'])
sns.lineplot('param_c__n_neighbors', 'mean_test_score',
             data=knn_results_p[knn_results_p['param_c__metric'] == 'manhattan'])
plt.legend(['Without Preprocessing', 'With Preprocessing'])
plt.savefig('plots/knn_comparison.png')
plt.show()

# %%
X_best_knn = X[best_rfecv_features]

# %% KNN Scorer and Time
results = cross_validate(knn_p, X_best_knn, y, scoring=scoring, cv=10)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% KNN HO
X_train, X_test, y_train, y_test = train_test_split(X_best_knn, y, test_size=0.2, random_state=random,
                                                    stratify=y)
knn_np.best_estimator_.fit(X_train, y_train)
print('Best Score Hold Out', knn_np.best_estimator_.score(X_test, y_test))

# %% RF CV approx params
param_grid = {
    # randomly sample numbers from 4 to 204 estimators
    'n_estimators': randint(11, 20),
    # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
    'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
    # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
    # 'min_samples_split': uniform(0.01, 0.199)
}

rf = RandomizedSearchCV(RandomForestClassifier(random_state=random, criterion='gini'), param_grid, cv=5,
                        n_jobs=-1, random_state=random, n_iter=100, verbose=True)

rf.fit(X[best_rfecv_features], y)

# %% RF CV NP
pipeline = Pipeline([('c', RandomForestClassifier(random_state=random, min_samples_split=0.01))])
n_estimators = np.arange(30, 35)
max_features = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
param_grid = dict(c__n_estimators=n_estimators, c__max_features=max_features)

rf_np = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
rf_np.fit(X, y)
print('Best Mean Score without Preprocessing', rf_np.best_score_, 'Model', rf.best_estimator_)
rf_np_results = pd.DataFrame(rf_np.cv_results_)
rf_np_results['param_c__max_features'] = list(
    map(lambda x: str(x * 100) + ' %', rf_np_results['param_c__max_features']))

sns.lineplot('param_c__n_estimators', 'mean_test_score', 'param_c__max_features', style='param_c__max_features',
             data=rf_np_results)
plt.savefig('plots/rf_comparison.png')
plt.show()

# %% RF CV P
pipeline = Pipeline([('s', StandardScaler()),
                     ('c', RandomForestClassifier(random_state=random, min_samples_split=0.01))])
param_grid = dict(c__n_estimators=n_estimators, c__max_features=max_features)

rf_p = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
rf_p.fit(X[best_rfecv_features], y)
print('Best Mean Score with Preprocessing', rf_p.best_score_, 'Model', rf.best_estimator_)

# %% RF Scorer and Time
results = cross_validate(rf_np, X[best_rfecv_features], y, scoring=scoring, cv=10, n_jobs=-1)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_Accuracy'].mean(), 'Precision',
      results['test_Precision'].mean(), 'Recall', results['test_Recall'].mean(), 'F1', results['test_F1'].mean())

# %% RF HO
X_train, X_test, y_train, y_test = train_test_split(X[best_rfecv_features], y, test_size=0.2, random_state=random)
rf_np.best_estimator_.fit(X_train, y_train)
print('Best Score Hold Out', rf_np.best_estimator_.score(X_test, y_test))

# %% MLP preprocessed
pipeline = Pipeline([('select', SelectKBest(chi2, k=1000)),
                     ('s', MinMaxScaler()),
                     ('c', MLPClassifier(random_state=random, max_iter=1000))])

hidden_layer_sizes = [(100,)]
activation = ['tanh', 'relu', 'logistic', 'identity']
param_grid = dict(c__hidden_layer_sizes=hidden_layer_sizes, c__activation=activation)

mlp = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    n_jobs=-1)

mlp.fit(X, y)
best_estimator_p = mlp.best_estimator_

print('Best Mean Score Without Preprocessing', mlp.best_score_, 'Model', mlp.best_estimator_)
mlp_results = pd.DataFrame(mlp.cv_results_)
# %% not pre processed


# %%

sns.barplot('param_c__hidden_layer_sizes', 'mean_test_score', 'param_c__activation', data=mlp_results)
plt.savefig('plots/mlp_comparison.png')
plt.show()

# %% RF Scorer and Time
# %% MLP CV NP
param_grid = {
    'hidden_layer_sizes': [(100,)],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
}

mlp = GridSearchCV(
    MLPClassifier(random_state=random), param_grid,
    cv=3,
    n_jobs=-1, verbose=True)

mlp.fit(X[best_rfecv_features], y)

print('Best Mean Score Without Preprocessing', mlp.best_score_, 'Model', mlp.best_estimator_)
mlp_results = pd.DataFrame(mlp.cv_results_)

# %% MLP CV P
classifier_pipeline = Pipeline([('s', preprocessing.MinMaxScaler()),
                                ('c', MLPClassifier(random_state=random))])

param_grid = dict(c__activation=['tanh', 'relu', 'logistic', 'identity'], c__hidden_layer_sizes=[(100,)])

mlp1 = GridSearchCV(classifier_pipeline, param_grid, cv=3,
                    n_jobs=-1, verbose=True)

mlp1.fit(X[best_rfecv_features], y)
best_estimator = mlp1.best_estimator_

print('Best Mean Score With Preprocessing', mlp1.best_score_, 'Model', mlp1.best_estimator_)
mlp1_results = pd.DataFrame(mlp1.cv_results_)

# %%

sns.barplot('param_mlpclassifier__hidden_layer_sizes', 'mean_test_score', 'param_mlpclassifier__activation', data=mlp1_results)
plt.savefig("plots/mlp_p_comparision.png")
plt.show()

plotdata = mlp_results[mlp_results['param_hidden_layer_sizes'] == (100,)]
temp = mlp1_results[mlp1_results['param_mlpclassifier__hidden_layer_sizes'] == (100,)]
temp = temp.rename(columns={'param_mlpclassifier__hidden_layer_sizes': 'param_hidden_layer_sizes',
                            'param_mlpclassifier__activation': 'param_activation'})
plotdata = plotdata.append(temp)
plotdata['param_hidden_layer_sizes'] = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']

sns.barplot('param_activation', 'mean_test_score', 'param_hidden_layer_sizes', data=plotdata)
plt.legend(['Without Preprocessing', 'With Preprocessing'])
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# # %% knn without pre processing comparison of features
# k = range(1, 60)
# metric = ['euclidean', 'chebyshev', 'manhattan']
# weights = ['uniform', 'distance']
# grid_search_dict = dict(n_neighbors=k, metric=metric, weights=weights)
# knn_np = GridSearchCV(KNeighborsClassifier(), grid_search_dict,
#                       cv=5, n_jobs=-1)
# knn_np.fit(X_best_k, y)
#
# best_estimator = knn_np.best_estimator_
# print('Best Mean Score without Preprocessing', knn_np.best_score_, 'Model',
#       best_estimator)
# knn_results = pd.DataFrame(knn_np.cv_results_)
# sns.lineplot('param_n_neighbors', 'mean_test_score', 'param_metric', style='param_metric', data=knn_results)
# plt.show()
#
# # %% knn with pre processing
# pipeline = Pipeline([('selector', SelectKBest(chi2)),
#                      ('scalar', preprocessing.MinMaxScaler()),
#                      ('c', KNeighborsClassifier(weights='distance'))])
# k = range(1, 60)
# metric = ['euclidean', 'chebyshev', 'manhattan']
# kbest = [1000]
# grid_search_dict = dict(c__n_neighbors=k, c__metric=metric, selector__k=kbest)
# knn_p = GridSearchCV(pipeline, grid_search_dict, cv=10, n_jobs=-1)
#
#
# def knn_p_plot(X):
#     knn_p.fit(X, y)
#     best_estimator = knn_p.best_estimator_
#     print('Best Mean Score with Preprocessing', knn_p.best_score_, 'Model', best_estimator)
#     knn_results = pd.DataFrame(knn_p.cv_results_)
#     sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric', style='param_c__metric',
#                  data=knn_results)
#     plt.show()  # TODO plot them together
#
#
# knn_p_plot(X_best_rfecv)
# knn_p_plot(X_best_k)
# knn_p_plot(X)
#
# # %% mlp with pca
# pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
#                      ('pca', PCA(0.95)),
#                      ('c', MLPClassifier(max_iter=1000, activation='relu'))])
# mlp_p = GridSearchCV(pipeline, param_grid={}, cv=3, n_jobs=-1)
# mlp_p.fit(X, y)
# mlp_best_estimator = mlp_p.best_estimator_
# print('Best Mean Score with Preprocessing', mlp_p.best_score_, 'Model',
#       mlp_best_estimator)
#
# # %% mlp with SelectKBest
# pipeline = Pipeline([('selector', SelectKBest(chi2)),
#                      ('scalar', preprocessing.MinMaxScaler()),
#                      # ('pca', PCA(0.99)),
#                      ('c', MLPClassifier(max_iter=1000, activation='relu'))])
# layers = [(100,)]
# k = [4000]
# grid_search_dict = dict(
#     c__hidden_layer_sizes=layers,
#     selector__k=k
# )
# mlp_p = GridSearchCV(pipeline, param_grid=grid_search_dict, cv=3, n_jobs=-1)
# mlp_p.fit(X, y)
# mlp_best_estimator = mlp_p.best_estimator_
# print('Best Mean Score with Preprocessing', mlp_p.best_score_, 'Model',
#       mlp_best_estimator)
#
# # %% mlp with best
# pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
#                      ('c', MLPClassifier())])
# activation = ['relu', 'tanh', 'logistic', 'sigmoid', 'softmax']
# grid_search_dict = dict(
#     c__activation=activation
# )
# mlp_p = GridSearchCV(pipeline, param_grid=grid_search_dict, cv=3, n_jobs=-1)
# mlp_p.fit(X_best_rfecv, y)
# mlp_best_estimator = mlp_p.best_estimator_
# print('Best Mean Score with Preprocessing', mlp_p.best_score_, 'Model', mlp_best_estimator)
# mlp_results = pd.DataFrame(mlp_p.cv_results_)
#
# sns.barplot('param_activation', 'mean_test_score', data=mlp_results)
# plt.show()
#
# # %%
# pipeline = Pipeline([('selector', SelectKBest(chi2)),
#                      ('scalar', preprocessing.MinMaxScaler()),
#                      ('c', RandomForestClassifier())])
# k = [2000]
#
# # %%
# score = 0
# k = 0
# for x in np.arange(1000, 6000, 10):
#     selector = SelectKBest(chi2, k=x)
#     X_best_k = selector.fit_transform(train.drop(['Class'], axis=1), train['Class'])
#     best_k_features = list(train.drop('Class', axis=1).columns[selector.get_support()])
#
#     scaler = preprocessing.MinMaxScaler().fit(train[best_k_features])
#     temp = cross_val_score(MLPClassifier(max_iter=1200, activation='relu'), scaler.transform(train[best_k_features]),
#                            train['Class'], cv=4).mean()
#     print(x)
#     if temp > score:
#         score = temp
#         k = x
#
# # %%
# test = pd.read_csv(dataset_path + "amazon_review_ID.shuf.tes.csv")
# sample_solution = pd.read_csv(dataset_path + "amazon_review_ID.shuf.sol.ex.csv")
#
# scaler = preprocessing.MinMaxScaler().fit(train[best_k_features])
# mlp = MLPClassifier(
#     max_iter=1000,
#     activation='relu',
# )
# mlp.fit(scaler.transform(train[best_k_features]), train['Class'])
# prediction = mlp.predict(scaler.transform(test[best_k_features]))
#
# sample_solution['Class'] = prediction
# sample_solution.to_csv("amazon/dataset/sol.csv", index=False)
#
# # %%
# test = pd.read_csv(dataset_path + "amazon_review_ID.shuf.tes.csv")
# sample_solution = pd.read_csv(dataset_path + "amazon_review_ID.shuf.sol.ex.csv")
#
# X_train = rf_ecv.transform(scaler_min_max.transform(X))
# X_test = rf_ecv.transform(scaler_min_max.transform(test.drop('ID', axis=1)))
#
# mlp = MLPClassifier(
#     hidden_layer_sizes=(1100, 1100, 300),
#     max_iter=1000,
#     activation='relu',
#     verbose=True
# )
# mlp.fit(X_train, y)
# prediction = mlp.predict(X_test)
#
# sample_solution['Class'] = prediction
# sample_solution.to_csv("amazon/dataset/sol.csv", index=False)
#
# # %%
# from sklearn.svm import SVC
#
# test = pd.read_csv(dataset_path + "amazon_review_ID.shuf.tes.csv")
# sample_solution = pd.read_csv(dataset_path + "amazon_review_ID.shuf.sol.ex.csv")
#
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X)
# X_test = scaler.transform(test.drop('ID', axis=1))
#
# svc = SVC()
#
# svc.fit(X_train, y)
# prediction = svc.predict(X_test)
#
# sample_solution['Class'] = prediction
# sample_solution.to_csv("amazon/dataset/sol.csv", index=False)
