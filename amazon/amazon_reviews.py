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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, validation_curve

plt.rcParams["patch.force_edgecolor"] = True

# %% load datasets
dataset_path = "amazon/dataset/"
train = pd.read_csv(dataset_path + "amazon_review_ID.shuf.lrn.csv")

X = train.drop(['Class', 'ID'], axis=1)
y = train['Class']

# %%
best_first_features = ["V289", "V448", "V821", "V1011", "V1295", "V1379", "V1397", "V6629", "V6924", "V6939", "V7468",
                       "V8058", "V8059", "V9200"]

# %% Select K Best
selector = SelectKBest(chi2, k=4000)
data_best_k = selector.fit_transform(X, y)
best_k_features = list(X.columns[selector.get_support()])


# %% Heatplot of found attributes


def plot_heatmap(df):
    correlation_matrix = df.corr().abs()
    sns.heatmap(correlation_matrix, linewidths=.5).get_figure()
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.show()


# %%
plot_heatmap(X[best_k_features])

# %%
plot_heatmap(X[best_first_features])

# %% knn pca
pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
                     ('pca', PCA()),
                     ('c', KNeighborsClassifier(weights='distance'))])

k = range(1, 60)
metric = ['euclidean', 'chebyshev', 'manhattan']
pca = [0.8, 0.85, 0.9, 0.95]

grid_search_dict = dict(c__n_neighbors=k, c__metric=metric, pca__n_components=pca)
knn_p = GridSearchCV(pipeline, grid_search_dict, cv=10, n_jobs=-1)
knn_p.fit(X[best_k_features], y)
best_estimator = knn_p.best_estimator_
print('Best Mean Score with Preprocessing', knn_p.best_score_, 'Model',
      best_estimator)

knn_results = pd.DataFrame(knn_p.cv_results_)
sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric',
             style='param_c__metric', data=knn_results)
plt.show()

# %%
pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
                     ('pca', PCA()),
                     ('c', KNeighborsClassifier(weights='distance'))])


# %% knn without pre processing comparison of features


def best_knn_np(attributes, plot=True):
    k = range(1, 60)
    metric = ['euclidean', 'chebyshev', 'manhattan']
    weights = ['uniform', 'distance']
    grid_search_dict = dict(n_neighbors=k, metric=metric, weights=weights)
    knn_np = GridSearchCV(KNeighborsClassifier(), grid_search_dict,
                          cv=10, n_jobs=-1)
    knn_np.fit(X[attributes], y)

    best_estimator = knn_np.best_estimator_
    print('Best Mean Score without Preprocessing', knn_np.best_score_, 'Model',
          best_estimator)
    if plot:
        knn_results = pd.DataFrame(knn_np.cv_results_)
        sns.lineplot('param_n_neighbors', 'mean_test_score', 'param_metric', style='param_metric', data=knn_results)
        plt.show()

    return best_estimator


best_esitmator_np = best_knn_np(best_k_features)

# %% knn with pre processing
pipeline = Pipeline([('selector', SelectKBest(chi2)),
                     ('scalar', preprocessing.MinMaxScaler()),
                     ('c', KNeighborsClassifier(weights='distance'))])
k = range(1, 60)
metric = ['euclidean', 'chebyshev', 'manhattan']
kbest = [1000]
grid_search_dict = dict(c__n_neighbors=k, c__metric=metric, selector__k=kbest)
knn_p = GridSearchCV(pipeline, grid_search_dict, cv=10, n_jobs=-1)
knn_p.fit(X, y)
best_estimator_p = knn_p.best_estimator_
print('Best Mean Score with Preprocessing', knn_p.best_score_, 'Model',
      best_estimator)

knn_results = pd.DataFrame(knn_p.cv_results_)
sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric',
             style='param_c__metric', data=knn_results)
plt.show()

# %% mlp with pca
pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
                     ('pca', PCA(0.95)),
                     ('c', MLPClassifier(max_iter=1000, activation='relu'))])
mlp_p = GridSearchCV(pipeline, param_grid={}, cv=3, n_jobs=-1)
mlp_p.fit(X, y)
mlp_best_estimator = mlp_p.best_estimator_
print('Best Mean Score with Preprocessing', mlp_p.best_score_, 'Model',
      mlp_best_estimator)

# %% mlp with
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
    data_best_k = selector.fit_transform(train.drop(['Class'], axis=1), train['Class'])
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
