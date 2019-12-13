# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV

from sklearn.pipeline import Pipeline

random = 47

train = pd.read_csv('breast/dataset/breast-cancer-diagnostic.shuf.lrn.csv')
train.columns = train.columns.str.strip()

# %% Pre-Processing
y = train['class']
X = train.drop(['ID', 'class'], axis=1)

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X)
X_min_max = min_max_scaler.transform(X)

# %% Count-Plot of Target
sns.countplot(y, label="Count")  # M = 212, B = 357
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
plt.show()

# %% Boxplot
plt.figure(figsize=(10, 10))
sns.boxplot(x="features", y="value", hue="class", data=data)
plt.xticks(rotation=90)
plt.show()

# %%
sns.jointplot(X['concavityWorst'], X['concavePointsWorst'], kind="regg", color="#ce1414")
plt.show()

# %%
plt.figure(figsize=(10, 10))
sns.swarmplot(x="features", y="value", hue="class", data=data)
plt.xticks(rotation=90)
plt.show()

# %%
drop_list1 = ['perimeterMean', 'radiusMean', 'compactnessMean', 'concavePointsMean', 'radiusStdErr', 'perimeterStdErr',
              'radiusWorst', 'perimeterWorst', 'compactnessWorst', 'concavePointsWorst', 'compactnessStdErr',
              'concavePointsStdErr', 'textureWorst', 'areaWorst']
X_best_features = X.drop(drop_list1, axis=1)  # do not modify x, we will use it later
X_best_features.head()

# %% Random Forest classifier without pre processing vs with pre processing
classifier_pipeline = Pipeline([('classifier', RandomForestClassifier(random_state=random, min_samples_split=0.01))])

param_grid = {
    'classifier__n_estimators': np.arange(1, 80),
    'classifier__max_features': [0.2, 0.3, 0.4, 0.5, 0.6],
}

rf = GridSearchCV(classifier_pipeline, param_grid,
                  cv=5,
                  n_jobs=-1)
rf.fit(X_best_features, y)
print('Best Mean Score Without Preprocessing', rf.best_score_, 'Model', rf.best_estimator_)

rf_results = pd.DataFrame(rf.cv_results_)
rf_results['param_classifier__max_features'] = list(
    map(lambda x: str(x * 100) + ' %', rf_results['param_classifier__max_features']))

sns.lineplot('param_classifier__n_estimators', 'mean_test_score',
             data=rf_results[rf_results['param_classifier__max_features'] == '50.0 %'])

# with pre processing
classifier_pipeline = Pipeline([('scaler', MinMaxScaler()),
                                ('classifier', RandomForestClassifier(random_state=random, min_samples_split=0.01))])

param_grid = {
    'classifier__n_estimators': np.arange(1, 80),
    'classifier__max_features': [0.2, 0.3, 0.4, 0.5, 0.6],
}

rf = GridSearchCV(classifier_pipeline, param_grid,
                  cv=5,
                  n_jobs=-1)
rf.fit(X_best_features, y)
print('Best Mean Score Without Preprocessing', rf.best_score_, 'Model', rf.best_estimator_)

rf_results = pd.DataFrame(rf.cv_results_)
rf_results['param_classifier__max_features'] = list(
    map(lambda x: str(x * 100) + ' %', rf_results['param_classifier__max_features']))
sns.lineplot('param_classifier__n_estimators', 'mean_test_score',
             data=rf_results[rf_results['param_classifier__max_features'] == '50.0 %'])
plt.legend(['Without Preprocessing', 'With Preprocessing'])
plt.show()

# %% Random Forest feature selection
# The "accuracy" scoring is proportional to the number of correct classifications
clf_rf_4 = RandomForestClassifier()
rf_ecv = RFECV(estimator=clf_rf_4, step=1, cv=10, scoring='accuracy', n_jobs=-1)  # 5-fold cross-validation
rf_ecv = rf_ecv.fit(X, y)

print('Optimal number of features :', rf_ecv.n_features_)
print('Best features :', X.columns[rf_ecv.support_])

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rf_ecv.grid_scores_) + 1), rf_ecv.grid_scores_)
plt.show()

# %% KNN classifier
pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()),
                     ('c', KNeighborsClassifier(weights='distance'))])
k = range(1, 20)
metric = ['euclidean', 'chebyshev', 'manhattan']

grid_search_dict = dict(c__n_neighbors=k, c__metric=metric)

knn_p = GridSearchCV(pipeline, grid_search_dict, cv=10, n_jobs=-1)
knn_p.fit(X_best_features, y)
best_estimator = knn_p.best_estimator_
print('Best Mean Score with Preprocessing', knn_p.best_score_, 'Model',
      best_estimator)

knn_results = pd.DataFrame(knn_p.cv_results_)
sns.lineplot('param_c__n_neighbors', 'mean_test_score', 'param_c__metric',
              style='param_c__metric', data=knn_results)
plt.show()

# %% mlp
mlp = cross_val_score(MLPClassifier(
    max_iter=1000,
    alpha=0.0001,
    activation='relu',
    solver='adam'), min_max_scaler.transform(X), y, cv=10, n_jobs=-1)

# %% Feature Selection
X_scale = min_max_scaler.transform(X)
mlp_activation = ['logistic', 'relu']
mlp_max_iter = range(1000, 5000, 1000)
mlp_alpha = [0.0001, 0.001, 0.01, 0.1]

mlp = GridSearchCV(MLPClassifier(solver='adam'), param_grid=[{'activation': mlp_activation},
                                                             {'alpha': mlp_alpha}, {'max_iter': mlp_max_iter}], cv=10)

mlp.fit(X_scale, y)
# max_iter=1000,alpha=0.0001,activation=relu
best_params = mlp.best_params_

# %% Kaggle Score 0.97647
X_scale = min_max_scaler.fit_transform(X)
test = pd.read_csv('breast/dataset/breast-cancer-diagnostic.shuf.tes.csv').drop('ID', axis=1)
test.columns = test.columns.str.strip()
sol = pd.read_csv('breast/dataset/breast-cancer-diagnostic.shuf.sol.ex.csv')

mlp = MLPClassifier(
    max_iter=1000,
    alpha=0.0001,
    activation='relu',
    solver='adam',
)
mlp.fit(X_scale, y)
prediction = mlp.predict(min_max_scaler.transform(test))

sol['class'] = prediction
sol.to_csv("breast/dataset/sol.csv", index=False)

# %%