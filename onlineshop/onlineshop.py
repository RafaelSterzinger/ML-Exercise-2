import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest as SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate


from onlineshop_utils import *

path = 'onlineshop/'
print(os.getcwd())
if os.getcwd().endswith('onlineshop'):
    path = ''
else:
    os.chdir(os.getcwd() + '\onlineshop')

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


scoring = ['accuracy', 'recall', 'precision', 'f1']
#scoring = 'recall'
refit = 'recall'

random_state = 42

#%% Data Analysis

sns.countplot(data['Revenue'])
plt.savefig("plots/revenue_boxplot.png")
plt.show()


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

print(attribute_scores.nlargest(12, 'Score').round(2))

# RandomTreeClassifier feature_importances_
k_feat_model = RandomForestClassifier(random_state=42)
k_feat_model.fit(data_encoded[numeric + categoric_encoded], data_encoded[target])

k_feat_values = pd.Series(k_feat_model.feature_importances_, index=data_encoded[numeric + categoric_encoded].columns).nlargest(10)
k_feat_values = k_feat_values.reindex(index=k_feat_values.index[::-1])
#sns.barplot(data=k_feat_values)
k_feat_values.plot(kind='barh')
# plt.subplots_adjust(left=0.15)
plt.savefig("plots/feature_selection_random_forest_classifier.png", bbox_inches ='tight')
plt.show

#%% correlation matrix
f, ax = plt.subplots(figsize=(10,8))
corr = data.corr().round(2)
sns.heatmap(corr, linewidths=1,# cmap=sns.diverging_palette(50, 150, as_cmap=True),
            annot=False)
# fix for cut off boxes
b, t = plt.ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
plt.ylim(b, t)  # update the ylim(bottom, top) values
plt.savefig("plots/heatmap.png")
plt.show()

#%% do not remove outliers, but instead add abnormality
# scores since the outliers seem to be the ones buying things
# increases accuracy by ~3%
print()
# TODO for preprocessing VS No-Preprocessing
#%% 1.1 k nearest neighbours                                                                                            Target Attribute comparison

knn_param_grid = {
    'n_neighbors' : np.arange(1, 20),
    'metric' : ['manhattan'], #['euclidean', 'chebyshev','manhattan'],
    'weights' : ['uniform'],
}

knn_top12_results, knn_wo_pre_best = attribute_comparison(KNeighborsClassifier(), data_encoded, top12_attributes, target, knn_param_grid, scoring, refit, 'Top 12')
knn_top7_results, trash = attribute_comparison(KNeighborsClassifier(),data_encoded,top7_attributes, target, knn_param_grid, scoring, refit, 'Top 7')
knn_top3_results, trash = attribute_comparison(KNeighborsClassifier(),data_encoded,top3_attributes, target, knn_param_grid, scoring, refit, 'Top 3')

print('Best Mean Score Without Preprocessing', knn_wo_pre_best.best_score_, 'Model', knn_wo_pre_best.best_estimator_)

knn_results = knn_top12_results.append(knn_top7_results).append(knn_top3_results)
#sns.lineplot('param_n_neighbors', 'mean_test_recall', 'param_metric', style='param_metric',data=knn_results)
sns.lineplot('param_n_neighbors', 'mean_test_recall', 'Attributes', style='Attributes',data=knn_results)
#sns.lineplot('param_n_neighbors', 'mean_test_accuracy', 'Attributes', style='Attributes',data=knn_results)

plt.ylim(0.1, 0.52)
plt.xticks(np.arange(0, 20, 2))
plt.savefig("plots/knn_wo_preprocessing.png")
plt.show()


#%% 1.1 k nearest neighbors With                                                                                        Min Max Scaling comparison -> User Preprocessing and Top 12 Attributes

# uncomment to compare Preprocessing vs No Preprocessing
#sns.lineplot('param_n_neighbors', 'mean_test_recall',data=knn_top12_results)                                            # no preprocessing

knn_param_grid_pipe = {
    'kneighborsclassifier__n_neighbors' : np.arange(1, 20),#31),
    'kneighborsclassifier__metric' : ['euclidean', 'chebyshev','manhattan'],#['manhattan'], #
    'kneighborsclassifier__weights' : ['uniform'] #['distance']#
    #'select__k': [3, 7, 12],
}
pipe = make_pipeline(preprocessing.MinMaxScaler(), KNeighborsClassifier())
'''    Pipeline([
    #('select', SelectKBest()),
    ('scaler', preprocessing.MinMaxScaler()),
    ('clf', KNeighborsClassifier())])'''

search = GridSearchCV(pipe, knn_param_grid_pipe, cv=5, n_jobs=-1, scoring=scoring, refit=refit)
search.fit(data_encoded[top12_attributes], data_encoded[target])


print('Best Mean Score With Preprocessing', search.best_score_, 'Model', search.best_estimator_)
knn_w_pre_best = search

search_results = pd.DataFrame(search.cv_results_)#.append(knn_top7.cv_results_).append(knn_top3.cv_results_)
sns.lineplot('param_kneighborsclassifier__n_neighbors', 'mean_test_recall',style='param_kneighborsclassifier__metric' ,data=search_results)
#plt.legend(['Without Preprocessing', 'With Preprocessing'])

plt.ylim(0.1, 0.52)
plt.xticks(np.arange(0, 20, 2))

plt.savefig("plots/knn_uniform_metric_comparison.png")
#plt.savefig("plots/knn_pre_vs_no_pre.png")
plt.show()

# %% 2.1 KNN Scorer and Time
best_estimator = knn_w_pre_best.best_estimator_

results = cross_validate(best_estimator, data_encoded[top12_attributes], data_encoded[target], scoring=scoring, cv=5)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_accuracy'].mean(), 'Precision',
      results['test_precision'].mean(), 'Recall', results['test_recall'].mean(), 'F1', results['test_f1'].mean())

# %% 3.1 KNN HO
X_train, X_test, y_train, y_test = train_test_split(data_encoded[top12_attributes], data_encoded[target], test_size=0.2, random_state=random_state,
                                                    stratify=data[target])
best_estimator.fit(X_train, y_train)
y_pred = best_estimator.predict(X_test)
print('Best Score Hold Out Recall', recall_score(y_test, y_pred))
print('Best Score Hold Out', best_estimator.score(X_test, y_test))

#%% 1.2 random Forest                                                                                                   max depth & n_estimators comparison

rfc=RandomForestClassifier(random_state=random_state)
rfc_param_grid = {
    'n_estimators': [10, 100, 300], #[10, 15, 20, 50],#
    'max_features': ['auto'],#, 'sqrt', 'log2'],
    'max_depth' : np.arange(5, 30, 2),
    'criterion' :['gini']#, ['entropy'] #
}

rfc_top12_results, rfc_wo_pre = attribute_comparison(rfc, data_encoded, top12_attributes, target, rfc_param_grid, scoring, refit, 'Top 12', cv=3)
#rfc_top7_results, trash = attribute_comparison(rfc,data_encoded,top7_attributes, target, rfc_param_grid, scoring, refit, 'Top 7')
#rfc_top3_results, trash = attribute_comparison(rfc,data_encoded,top3_attributes, target, rfc_param_grid, scoring, refit, 'Top 3')

print('Best Mean Score Without Preprocessing', rfc_wo_pre.best_score_, 'Model', rfc_wo_pre.best_estimator_)

rfc_results = rfc_top12_results#.append(rfc_top7_results).append(rfc_top3_results)

for n_estim in rfc_param_grid['n_estimators']:
    print(str(n_estim))
    rfc_results.loc[rfc_results['param_n_estimators'] == n_estim, 'Estimators'] = '{' + str(n_estim) + '}' # cannot be an integer
    #rfc_results['Estimators'] =  str(rfc_results['param_n_estimators'][2])
sns.lineplot('param_max_depth', 'mean_test_recall', 'Estimators', style='Estimators',
             data=rfc_top12_results, legend='full')

plt.ylim(0.545, 0.635)

plt.savefig("plots/rfc_n_estimators_comparison.png")
plt.show()


#%% 1.2 random forest With                                                                                              Min Max Scaling comparison -> Pre Processing makes it worse

sns.lineplot('param_max_depth', 'mean_test_recall', data=rfc_results[rfc_results['param_n_estimators']==100])                                            # no preprocessing

rfc_param_grid_pipe = {
    'randomforestclassifier__n_estimators': [100],#[100, 150, 200],
    'randomforestclassifier__max_features': ['auto'],#, 'sqrt', 'log2'],
    'randomforestclassifier__max_depth' : np.arange(5, 30, 2),
    'randomforestclassifier__criterion' :['gini']#, 'entropy']
}

pipe = make_pipeline(preprocessing.MinMaxScaler(), RandomForestClassifier(random_state=random_state))

search = GridSearchCV(pipe, rfc_param_grid_pipe, cv=5, n_jobs=-1, scoring=scoring, refit=refit)
search.fit(data_encoded[top12_attributes], data_encoded[target])

print('Best Mean Score With Preprocessing', search.best_score_, 'Model', search.best_estimator_)
rfc_best_estimator = search

search_results = pd.DataFrame(search.cv_results_)#.append(knn_top7.cv_results_).append(knn_top3.cv_results_)
sns.lineplot('param_randomforestclassifier__max_depth', 'mean_test_recall',data=search_results)
plt.legend(['Without Preprocessing', 'With Preprocessing'])

plt.ylim(0.545, 0.635)

plt.savefig("plots/rfc_preprocessing_comparison.png")
plt.show()

# %% 2.2 RFC Scorer and Time
best_estimator = rfc_wo_pre.best_estimator_

results = cross_validate(best_estimator, data_encoded[top12_attributes], data_encoded[target], scoring=scoring, cv=5)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_accuracy'].mean(), 'Precision',
      results['test_precision'].mean(), 'Recall', results['test_recall'].mean(), 'F1', results['test_f1'].mean())

# %% 3.2 RFC HO
X_train, X_test, y_train, y_test = train_test_split(data_encoded[top12_attributes], data_encoded[target], test_size=0.2, random_state=random_state,
                                                    stratify=data[target])
best_estimator.fit(X_train, y_train)
y_pred = best_estimator.predict(X_test)
print('Best Score Hold Out Recall', recall_score(y_test, y_pred))
print('Best Score Hold Out', best_estimator.score(X_test, y_test))


#%% 1.3 MLP CV approx params        DEPRECATED
'''mlp_param_grid = {
    'hidden_layer_sizes': [(3, 4, 3), (7, 5, 3), (3, 3, 7)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.01, 0.001, 0.0001]
}

mlp = RandomizedSearchCV(MLPClassifier(max_iter=5000, random_state=random_state), mlp_param_grid, cv=2,
                         n_jobs=-1, random_state=random_state, verbose=True)#,scoring=scoring, refit= refit)

mlp.fit(data_encoded[top12_attributes], data_encoded[target])
print('Best Mean Score Without Preprocessing', mlp.best_score_, 'Model', mlp.best_estimator_)


best_estimator = mlp.best_estimator_
'''


#%% 1.3 MLP CV NP       RECALL ZERO
mlp_param_grid = {
    'hidden_layer_sizes': [(7, 3, 5, 5, 10)],#[(3, 4, 3), (7, 5, 3), (3, 3, 7)],
    'activation': ['tanh', 'relu', 'logistic', 'identity'],
}

mlp = GridSearchCV(
    MLPClassifier(alpha=0.001, solver='adam', learning_rate='constant', max_iter=3000, random_state=random_state), mlp_param_grid,
    cv=3,
    verbose=True,
    n_jobs=-1, scoring = 'recall')#, scoring=scoring, refit=refit)

mlp.fit(data_encoded[top12_attributes], data_encoded[target])
#best_estimator = mlp.best_estimator_

print('Best Mean Score Without Preprocessing', mlp.best_score_, 'Model', mlp.best_estimator_)
mlp_results = pd.DataFrame(mlp.cv_results_)

#sns.barplot('param_hidden_layer_sizes', 'mean_test_score'#recall'
 #           , 'param_activation', data=mlp_results)
#plt.show()

# %% 1.3 MLP CV Preprocessing
solver = ['sgd', 'adam']
for sol in solver:
    classifier_pipeline = make_pipeline(preprocessing.MinMaxScaler(),
                                        MLPClassifier(alpha=0.001, solver=sol, learning_rate='adaptive', max_iter=3000,  # solver = 'sgd' performs much worse
                                                      random_state=random_state))

    param_grid = {
        'mlpclassifier__hidden_layer_sizes': [(3, 4, 3), (7, 5, 3, 7), (7, 3, 5, 5, 10), (15, 30, 15), (30, 30, 30)],#[(15,15,15),(30, 15, 30), (15, 30, 15), (30, 30, 30)], #[(3, 4, 3), (7, 5, 3), (3, 3, 7)], #
        'mlpclassifier__activation': ['tanh', 'relu', 'logistic', 'identity'],
    }

    mlp1 = GridSearchCV(classifier_pipeline, param_grid, cv=3,
                        n_jobs=-1, verbose=True, scoring='recall')

    mlp1.fit(data_encoded[top12_attributes],data_encoded[target])
    best_estimator = mlp1.best_estimator_

    print('Best Mean Score With Preprocessing', mlp1.best_score_, 'Model', mlp1.best_estimator_, 'Solver', sol)
    mlp1_results = pd.DataFrame(mlp1.cv_results_)

    sns.barplot('param_mlpclassifier__hidden_layer_sizes', 'mean_test_score', 'param_mlpclassifier__activation', data=mlp1_results)
    #plt.ylim(0.8, 0.9)
    plt.ylim(0, 0.7)
    plt.savefig("plots/mlp_solver_comparison" + sol + ".png")
    plt.show()
best_estimator = mlp1.best_estimator_


# %% 2.3 MLP Scorer and Time

results = cross_validate(best_estimator, data_encoded[top12_attributes], data_encoded[target], scoring=scoring, cv=5)
print('Time', results['fit_time'].mean(), 'Accuracy', results['test_accuracy'].mean(), 'Precision',
      results['test_precision'].mean(), 'Recall', results['test_recall'].mean(), 'F1', results['test_f1'].mean())

# %% 3.3 MLP HO
X_train, X_test, y_train, y_test = train_test_split(data_encoded[top12_attributes], data_encoded[target], test_size=0.2, random_state=random_state,
                                                    stratify=data[target])
best_estimator.fit(X_train, y_train)
y_pred = best_estimator.predict(X_test)
print('Best Score Hold Out Recall', recall_score(y_test, y_pred))
print('Best Score Hold Out', best_estimator.score(X_test, y_test))
