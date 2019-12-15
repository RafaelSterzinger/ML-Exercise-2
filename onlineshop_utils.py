import numpy as np
import pandas as pd

# returns a train set and a validation set
# validation
from sklearn.model_selection import GridSearchCV


def train_validate_test_split(df, train_percent=.8, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate#, test

def attribute_comparison(model, data, attributes, target, param_grid, scoring, refit, attributename, cv=10):
    search = GridSearchCV(model, param_grid=param_grid,
                             cv=cv, n_jobs=-1, scoring=scoring, refit=refit)
    search.fit(data[attributes], data[target])

    search_results = pd.DataFrame(search.cv_results_)  # .rename('top12')
    search_results['Attributes'] = attributename

    return search_results, search

