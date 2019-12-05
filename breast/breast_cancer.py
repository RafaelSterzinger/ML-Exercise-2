# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
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

random = 47

train = pd.read_csv('breast/dataset/breast-cancer-diagnostic.shuf.lrn.csv')
train.columns = train.columns.str.strip()

# %% Pre-Processing
y = train['class']
X = train.drop(['ID', 'class'], axis=1)

X.describe()
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(X)

# %% Count-Plot of Target
sns.countplot(y, label="Count")  # M = 212, B = 357
plt.show()

# %%
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
X_scale = min_max_scaler.transform(X)
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
