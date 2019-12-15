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
import sklearn


dataset_path = "amazon/dataset/"
train = pd.read_csv(dataset_path + "amazon_review_ID.shuf.lrn.csv")

X = train.drop(['Class', 'ID'], axis=1)
y = train['Class']

grid_search_dict = dict(

)

select =sklearn.feature_selection.variance_threshold.VarianceThreshold()
select.fit(X,y)
X = select.transform(X)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

model = cross_validate(sklearn.svm.LinearSVC(),X,y,cv=3,verbose=True,n_jobs=-1)


# %%
from sklearn.svm import SVC

test = pd.read_csv(dataset_path + "amazon_review_ID.shuf.tes.csv")
sample_solution = pd.read_csv(dataset_path + "amazon_review_ID.shuf.sol.ex.csv")

X_train = scaler.fit_transform(X)
X_test = scaler.transform(test.drop('ID', axis=1))

svc = SVC()

svc.fit(X_train, y)
prediction = svc.predict(X_test)

sample_solution['Class'] = prediction
sample_solution.to_csv("amazon/dataset/sol.csv", index=False)
