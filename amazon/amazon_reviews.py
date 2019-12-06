import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

plt.rcParams["patch.force_edgecolor"] = True

# %% load datasets
dataset_path = "amazon/dataset/"
data = pd.read_csv(dataset_path + "amazon_review_ID.shuf.lrn.csv").drop('ID',axis=1)

# %%
# correlation_matrix = data.corr().abs()
# sns.heatmap(correlation_matrix, linewidths=.5).get_figure()
# plt.savefig(path + "heatmap_numerical.png")
# plt.show()

# %%
best_first_features = ["V289", "V448", "V821", "V1011", "V1295", "V1379", "V1397", "V6629", "V6924", "V6939", "V7468",
              "V8058", "V8059", "V9200"]

#%%
selector = SelectKBest(chi2, k=3000)
data_best_k = selector.fit_transform(data.drop('Class', axis=1),data['Class'])
best_k_features = list(data.drop('Class', axis=1).columns[selector.get_support()])

#%% Heatplot of found attributes
best_k_features.append('Class')
correlation_matrix = data[best_k_features].corr().abs()
sns.heatmap(correlation_matrix, linewidths=.5).get_figure()
plt.show()
best_k_features.remove('Class')

#%%
best_first_features.append('Class')
correlation_matrix = data[best_first_features].corr().abs()
sns.heatmap(correlation_matrix, linewidths=.5).get_figure()
plt.show()
best_first_features.remove('Class')


#%% knn comparison of features
k = range(1,30)
best_first_results = []
best_k_results = []
for i in k:
    classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(), KNeighborsClassifier(i))
    result = np.mean(cross_val_score(classifier_pipeline, data[best_first_features], data['Class'], cv=10))
    best_first_results.append(result)
    result = np.mean(cross_val_score(classifier_pipeline, data[best_k_features], data['Class'], cv=10))
    best_k_results.append(result)

plt.plot(k, best_first_results, '--', label='best first features')
plt.plot(k, best_k_results, '--', label='best k features')
plt.xticks(k)
plt.xlabel('# neighbours (k)')
plt.ylabel('fitted')
plt.legend()
plt.show()

# %%
scaler = preprocessing.MinMaxScaler().fit(data[best_k_features])
score = cross_val_score(MLPClassifier(max_iter=1000,activation='relu'),scaler.transform(data[best_k_features]),data['Class'],cv=5).mean()

#%%
score = 0
k = 0
for x in np.arange(1000,6000,10):
    selector = SelectKBest(chi2, k=x)
    data_best_k = selector.fit_transform(data.drop(['Class'], axis=1),data['Class'])
    best_k_features = list(data.drop('Class', axis=1).columns[selector.get_support()])

    scaler = preprocessing.MinMaxScaler().fit(data[best_k_features])
    temp = cross_val_score(MLPClassifier(max_iter=1200, activation='relu'), scaler.transform(data[best_k_features]),
                            data['Class'], cv=4).mean()
    print(x)
    if temp > score:
        score = temp
        k = x

#%%
test = pd.read_csv(dataset_path + "amazon_review_ID.shuf.tes.csv")
sample_solution = pd.read_csv(dataset_path + "amazon_review_ID.shuf.sol.ex.csv")

scaler = preprocessing.MinMaxScaler().fit(data[best_k_features])
mlp = MLPClassifier(
    max_iter=1000,
    activation='relu',
)
mlp.fit(scaler.transform(data[best_k_features]), data['Class'])
prediction = mlp.predict(scaler.transform(test[best_k_features]))

sample_solution['Class'] = prediction
sample_solution.to_csv("amazon/dataset/sol.csv", index=False)
