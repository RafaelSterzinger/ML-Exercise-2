import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest as SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from onlineshop_utils import *

path = 'onlineshop/'
print(os.getcwd())
if os.getcwd().endswith('onlineshop'):
    path = ''

plt.rcParams["patch.force_edgecolor"] = True

data = pd.read_csv(path + 'dataset/online_shoppers_intention.csv') # 'onlineshop/dataset/online_shoppers_intention.csv'
# date_encoded ... data with OHE

# split 20 percent randomly into validation for Holdout Method
# data, validate = train_validate_test_split(data)


target = 'Revenue'

numeric  = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
categoric= ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']#, 'Revenue']
categoric_encoded = [] # contains Month_Feb, ..., Month_Dec, OperatingSystems_1, ... OperatingSystems_8
                       # no data available for Jan and April

top3_attributes = ['ProductRelated_Duration', 'PageValues','Administrative_Duration']
top7_attributes = top3_attributes + ['Informational_Duration', 'ProductRelated', 'Administrative', 'Informational']
top12_attributes = top7_attributes +  ['BounceRates', 'ExitRates', 'PageValues', 'Month_Nov', 'TrafficType' ]          # evaluated from correlation matrix

#%% Data Analasys

# 84.5% (10,422) were negative class samples that did not end with shopping, and the rest (1908) were positive class samples ending with shopping.
# thus it is more important to not identify Buyers as Non-Buyers i.e. one would rather give more people more coupons than potentially lose customers
# One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same.

# Recall is important (TP / (TP + FN). We choose this as there is a high cost with FN and because our data is skewed

# Precision is important if cost of FP is high (e.g. email spam detection: might lose important messages)

# Attributes
print(list(data.columns)[:])

# missing values only for the Kaggle Dataset (not for the archive one)
data.isnull().sum()

#%% pre processing

# One Hot Encoding instead of Label Encoding so that higher numbers are not biased
categoric_encoded = []
for cat_attribute in categoric:
    if cat_attribute != 'TrafficType':
        distinct_category_values = list(set(data[cat_attribute]))
        print(cat_attribute, distinct_category_values)
        for category_value in distinct_category_values:
            categoric_encoded.append(cat_attribute + '_'+ str(category_value))
    else:
        categoric_encoded.append('TrafficType')

print(len(categoric_encoded) ,categoric_encoded)

categoric_without_Traffictype = categoric.copy()
categoric_without_Traffictype.remove('TrafficType')
data_encoded = pd.get_dummies(data, columns=categoric_without_Traffictype)

print(len(data_encoded.columns),data_encoded.columns)

# all categories correctly encoded
for x in categoric_encoded:
    print(data_encoded[x][1])

#%% feature selection to reduce overfitting, improve accuracy and reduce training time

# sklearn selectkbest
best_features = SelectKBest(score_func=chi2, k=10)
fit = best_features.fit(data_encoded[numeric + categoric_encoded], data_encoded[target])

df_fit_scores = pd.DataFrame(fit.scores_)
df_attribute_columns = pd.DataFrame(data_encoded[numeric + categoric_encoded].columns)
attribute_scores = pd.concat([df_attribute_columns, df_fit_scores], axis=1)
attribute_scores.columns = ['Atribute', 'Score']

print(attribute_scores.nlargest(12, 'Score'))

# correlation matrix
f, ax = plt.subplots(figsize=(10,8))
corr = data.corr().round(2)
sns.heatmap(corr,# cmap=sns.diverging_palette(50, 150, as_cmap=True),
            annot=True)

plt.show()

#%% do not remove outliers, but instead add abnormality
# scores since the outliers seem to be the ones buying things
# increases accuracy by ~3%

#%% 1.1 k nearest neighbours Parameter iterieren
k = np.arange(1, 31)
metric = ['euclidean', 'chebyshev','manhattan']
scoring = ['accuracy', 'recall']
#scoring = 'recall'
refit = 'recall'

knn_param_grid = {
    'n_neighbors' : np.arange(1, 31),
    'metric' : ['euclidean', 'chebyshev','manhattan'],
    'weights' : ['uniform'],
}

knn = GridSearchCV(KNeighborsClassifier(), param_grid=knn_param_grid,
                   cv=10, n_jobs=-1, scoring=scoring, refit=refit)
knn.fit(data_encoded[top12_attributes], data_encoded[target])
best_estimator_NP = knn.best_estimator_


knn_results = pd.DataFrame(knn.cv_results_)
sns.lineplot('param_n_neighbors', 'mean_test_recall', 'param_metric', style='param_metric',data=knn_results)
plt.show()

#%%
model = RandomForestClassifier(bootstrap=True,
                                                    class_weight=None,
                                                    criterion='gini',
                                                    max_depth= 9,
                                                    max_features='auto',
                                                    n_estimators=300,
                                                    n_jobs=None,)
'''
model = RandomForestClassifier(bootstrap=True,
                                                    class_weight=None,
                                                    criterion='gini',
                                                    max_depth= 9,
                                                    max_features='auto',
                                                    max_leaf_nodes=None,
                                                    min_impurity_decrease=0.0,
                                                    min_impurity_split=None,
                                                    min_samples_leaf=1,
                                                    min_samples_split=2,
                                                    min_weight_fraction_leaf=0.0,
                                                    n_estimators=300,
                                                    n_jobs=None,)
'''
classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(), model)

'''
scores3 = cross_val_score(classifier_pipeline, data_encoded[top3_attributes], data_encoded[target], cv=10).mean()
print(scores3)

scores7 = cross_val_score(classifier_pipeline, data_encoded[top7_attributes], data_encoded[target], cv=10).mean()
print(scores7)
'''
scores12 = cross_val_score(classifier_pipeline, data_encoded[top12_attributes], data_encoded[target], cv=10, scoring='recall').mean()
print(scores12)

#%% Grid Search RandomForest
rfc=RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [250, 300, 350],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [7,8, 9, 10],
    'criterion' :['gini', 'entropy']
}

randomForestGrid = GridSearchCV(estimator=rfc, param_grid=param_grid,
                   cv=10, n_jobs=-1, scoring=scoring, refit=refit)

randomForestGrid.fit(data_encoded[top12_attributes], data_encoded[target])

best_estimator_RFC = rfc.base_estimator_

rfc_results = pd.DataFrame(randomForestGrid.cv_results_)
sns.lineplot('param_max_depth', 'mean_test_score', 'param_n_estimators', style='param_n_estimators',data=rfc_results)
plt.show()


