import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve

random = 123
plt.rcParams["patch.force_edgecolor"] = True

# %% load datasets
dataset_path = "amazon/dataset/"
train = pd.read_csv(dataset_path + "amazon_review_ID.shuf.lrn.csv")

X = train.drop(['Class', 'ID'], axis=1)
y = train['Class']

# %% plot target
plt.figure(figsize=(10, 6))
chart = sns.countplot(y, label="Count")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
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


# %% knn cv np, k and metrics
pipeline = Pipeline([('c', KNeighborsClassifier(weights='distance'))])

k = range(1, 60)
metric = ['euclidean', 'chebyshev', 'manhattan']

grid_search_dict = dict(c__n_neighbors=k, c__metric=metric)
knn_p = GridSearchCV(pipeline, grid_search_dict, cv=5, n_jobs=-1)
knn_p.fit(X, y)
best_estimator = knn_p.best_estimator_
print('Best Mean Score without Preprocessing', knn_p.best_score_, 'Model',
      best_estimator)
knn_results_np = pd.DataFrame(knn_p.cv_results_)

# %% knn cv p, k and metrics
classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(), KNeighborsClassifier(weights='distance'))
knn = GridSearchCV(classifier_pipeline, dict(kneighborsclassifier__n_neighbors=k, kneighborsclassifier__metric=metric),
                   cv=5, n_jobs=-1)

knn.fit(X[best_rfecv_features], y)

print('Best Mean Score With Preprocessing', knn.best_score_, 'Model', knn.best_estimator_)
knn_results_p = pd.DataFrame(knn.cv_results_)

# %% plot results
sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric', style='param_c__metric',
             data=knn_results_np['param_metric'] == 'manhattan')
sns.lineplot('param_kneighborsclassifier__n_neighbors', 'mean_test_score',
             data=knn_results_p[knn_results_p['param_kneighborsclassifier__metric'] == 'manhattan'])
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %% knn without pre processing comparison of features
k = range(1, 60)
metric = ['euclidean', 'chebyshev', 'manhattan']
weights = ['uniform', 'distance']
grid_search_dict = dict(n_neighbors=k, metric=metric, weights=weights)
knn_np = GridSearchCV(KNeighborsClassifier(), grid_search_dict,
                      cv=5, n_jobs=-1)
knn_np.fit(X_best_k, y)

best_estimator = knn_np.best_estimator_
print('Best Mean Score without Preprocessing', knn_np.best_score_, 'Model',
      best_estimator)
knn_results = pd.DataFrame(knn_np.cv_results_)
sns.lineplot('param_n_neighbors', 'mean_test_score', 'param_metric', style='param_metric', data=knn_results)
plt.show()

# %% knn with pre processing
pipeline = Pipeline([('selector', SelectKBest(chi2)),
                     ('scalar', preprocessing.MinMaxScaler()),
                     ('c', KNeighborsClassifier(weights='distance'))])
k = range(1, 60)
metric = ['euclidean', 'chebyshev', 'manhattan']
kbest = [1000]
grid_search_dict = dict(c__n_neighbors=k, c__metric=metric, selector__k=kbest)
knn_p = GridSearchCV(pipeline, grid_search_dict, cv=10, n_jobs=-1)


def knn_p_plot(X):
    knn_p.fit(X, y)
    best_estimator = knn_p.best_estimator_
    print('Best Mean Score with Preprocessing', knn_p.best_score_, 'Model', best_estimator)
    knn_results = pd.DataFrame(knn_p.cv_results_)
    sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric', style='param_c__metric',
                 data=knn_results)
    plt.show()  # TODO plot them together


knn_p_plot(X_best_rfecv)
knn_p_plot(X_best_k)
knn_p_plot(X)

# %% mlp with pca
pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
                     ('pca', PCA(0.95)),
                     ('c', MLPClassifier(max_iter=1000, activation='relu'))])
mlp_p = GridSearchCV(pipeline, param_grid={}, cv=3, n_jobs=-1)
mlp_p.fit(X, y)
mlp_best_estimator = mlp_p.best_estimator_
print('Best Mean Score with Preprocessing', mlp_p.best_score_, 'Model',
      mlp_best_estimator)

# %% mlp with SelectKBest
pipeline = Pipeline([('selector', SelectKBest(chi2)),
                     ('scalar', preprocessing.MinMaxScaler()),
                     # ('pca', PCA(0.99)),
                     ('c', MLPClassifier(max_iter=1000, activation='relu'))])
layers = [(100,)]
k = [4000]
grid_search_dict = dict(
    c__hidden_layer_sizes=layers,
    selector__k=k
)
mlp_p = GridSearchCV(pipeline, param_grid=grid_search_dict, cv=3, n_jobs=-1)
mlp_p.fit(X, y)
mlp_best_estimator = mlp_p.best_estimator_
print('Best Mean Score with Preprocessing', mlp_p.best_score_, 'Model',
      mlp_best_estimator)

# %% mlp with best
pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
                     ('c', MLPClassifier())])
activation = ['relu', 'tanh', 'logistic', 'sigmoid', 'softmax']
grid_search_dict = dict(
    c__activation=activation
)
mlp_p = GridSearchCV(pipeline, param_grid=grid_search_dict, cv=3, n_jobs=-1)
mlp_p.fit(X_best_rfecv, y)
mlp_best_estimator = mlp_p.best_estimator_
print('Best Mean Score with Preprocessing', mlp_p.best_score_, 'Model', mlp_best_estimator)
mlp_results = pd.DataFrame(mlp_p.cv_results_)

sns.barplot('param_activation', 'mean_test_score', data=mlp_results)
plt.show()

# %%
pipeline = Pipeline([('selector', SelectKBest(chi2)),
                     ('scalar', preprocessing.MinMaxScaler()),
                     ('c', RandomForestClassifier())])
k = [2000]

# %%
score = 0
k = 0
for x in np.arange(1000, 6000, 10):
    selector = SelectKBest(chi2, k=x)
    X_best_k = selector.fit_transform(train.drop(['Class'], axis=1), train['Class'])
    best_k_features = list(train.drop('Class', axis=1).columns[selector.get_support()])

    scaler = preprocessing.MinMaxScaler().fit(train[best_k_features])
    temp = cross_val_score(MLPClassifier(max_iter=1200, activation='relu'), scaler.transform(train[best_k_features]),
                           train['Class'], cv=4).mean()
    print(x)
    if temp > score:
        score = temp
        k = x

# %%
test = pd.read_csv(dataset_path + "amazon_review_ID.shuf.tes.csv")
sample_solution = pd.read_csv(dataset_path + "amazon_review_ID.shuf.sol.ex.csv")

scaler = preprocessing.MinMaxScaler().fit(train[best_k_features])
mlp = MLPClassifier(
    max_iter=1000,
    activation='relu',
)
mlp.fit(scaler.transform(train[best_k_features]), train['Class'])
prediction = mlp.predict(scaler.transform(test[best_k_features]))

sample_solution['Class'] = prediction
sample_solution.to_csv("amazon/dataset/sol.csv", index=False)

# %%
test = pd.read_csv(dataset_path + "amazon_review_ID.shuf.tes.csv")
sample_solution = pd.read_csv(dataset_path + "amazon_review_ID.shuf.sol.ex.csv")

X_train = rf_ecv.transform(scaler_min_max.transform(X))
X_test = rf_ecv.transform(scaler_min_max.transform(test.drop('ID', axis=1)))

mlp = MLPClassifier(
    hidden_layer_sizes=(1100, 1100, 300),
    max_iter=1000,
    activation='relu',
    verbose=True
)
mlp.fit(X_train, y)
prediction = mlp.predict(X_test)

sample_solution['Class'] = prediction
sample_solution.to_csv("amazon/dataset/sol.csv", index=False)
