# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

plt.rcParams["patch.force_edgecolor"] = True

data = pd.read_csv('iris/dataset/iris.data', names=['sep_length', 'sep_width', 'pet_length', 'pet_width', 'type'])
numeric = ['sep_length', 'sep_width', 'pet_length', 'pet_width']
target = 'type'

# %%
data.describe()
correlation_matrix = data[numeric].corr().round(2)
sns.heatmap(correlation_matrix, linewidths=1, annot=True)
# fix for cut off boxes
b, t = plt.ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
plt.ylim(b, t)  # update the ylim(bottom, top) values
plt.show()

# %%
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

# %% Best K
knn = GridSearchCV(KNeighborsClassifier(), [{'weights': ['uniform','distance']}, {'n_neighbors': np.arange(1, 100)},
                                            {'metric': ['euclidean', 'manhattan', 'chebyshev']}], cv=10)
knn.fit(data[numeric],data[target])

# %% KNN CV NP

scores = cross_val_score(knn.best_estimator_, data[numeric], data[target], cv=10).mean()

# %% KNN HO NP
X_train, X_test, y_train, y_test = train_test_split(data[numeric], data[target], test_size=0.2, random_state=1,
                                                    stratify=data[target])

# %%
classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(), KNeighborsClassifier())

scores = cross_val_score(classifier_pipeline, data[numeric], data[target], cv=10).mean()

# scores = cross_val_score(DecisionTreeClassifier(), data[numeric], data[target], cv=10).mean()

# scores = cross_val_score(MLPClassifier(max_iter=1000), data[numeric], data[target], cv=10).mean()
