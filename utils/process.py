import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils.validation import check_is_fitted

def tidy_features_type():
    "create csv containing predetermined features types"
    # categorical
    cat_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
                    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                    'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                    'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
                    'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
                    'SaleType', 'SaleCondition'
                    ]
    # continuous
    cont_features = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'MasVnrArea', 'BsmtFinSF1',
                     'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                     'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                     'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
                     ]

    # process dates
    dates = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']

    max_length = max(len(cat_features), len(cont_features), len(dates))

    def padded_list(x, max_length=max_length):
        padded = x + [''] * (max_length - len(x))
        return padded

    all_features = {'cat': padded_list(cat_features),
                    'cont': padded_list(cont_features),
                    'dates': padded_list(dates)}

    all_features = pd.DataFrame(all_features)
    #all_features.to_csv("features_types.csv", index=False)

    def filter_empy_strings(x):
        return [el for el in x if el]

    cat_features = filter_empy_strings(all_features['cat'].tolist())
    cont_features = filter_empy_strings(all_features['cont'].dropna().tolist())
    dates = filter_empy_strings(all_features['dates'].dropna().tolist())

    return cat_features, cont_features, dates


def convert_time_since(feature, X, reference='date_sold'):
    "convert to time difference in months"

    days_in_month = 30.44
    return (X[reference] - X[feature]).dt.days / days_in_month


def transform_dates(X, dates):
    format_date = lambda x: f"{x[0]}{str(x[1]).zfill(2)}"
    X['date_sold'] = pd.to_datetime(X[['YrSold', 'MoSold']].apply(format_date, 1), format="%Y%m")

    to_dt = [c for c in dates if c not in ['YrSold', 'MoSold']]
    for d in to_dt:
        X[d] = pd.to_datetime(X[d], format='%Y')

    X['time_since_built'] = convert_time_since('YearBuilt', X)
    X['time_since_remod'] = convert_time_since('YearRemodAdd', X)
    X['time_since_garage'] = convert_time_since('GarageYrBlt', X)

    return X


def determine_categories(X, cat_feat, fill_value):
    """
    Helper function used in conjunction with sklearn.preprocessing.OneHotEncoder
    to create categories informing OHE'ing of features

    Returns
    -------
        list of unique  with "missing" category added if nulls in features
    """

    if (X[cat_feat].isnull().sum() > 0) and (fill_value == 'missing'):
        return [X[cat_feat].unique().tolist() + ['missing']]
    else:
        return [X[cat_feat].unique().tolist()]

class TargetTransform(BaseEstimator, TransformerMixin):
    def __init__(self, transform_function, inverse_transform_function):
        self.transform_function = transform_function
        self.inverse_transform_function = inverse_transform_function

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        X = self.transform_function(X)
        return X

    def inverse_transform(self, X):
        X = self.inverse_transform_function(X)
        return X


class VarianceThresh(TransformerMixin):
    """ Variance threshold transformer

    Returns
    -------
        pd.DataFrame with column names as used in input dataframe

    Note
    ----
    As far as I know, there is currently no way to do this out of the box
    with sklearn_pandas.DataFrameMapper

    """

    def __init__(self, threshold=0.):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.selector = VarianceThreshold(self.threshold)
        self.selector.fit(X)
        self.column_names = X.columns[self.selector.get_support(indices=True)]
        return self

    def transform(self, X):
        check_is_fitted(self, 'selector')

        return pd.DataFrame(self.selector.transform(X), columns=self.column_names)

    def get_feature_names(self):
        return self.column_names


class PCAColumns(TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.reductor = PCA(self.n_components)
        self.reductor.fit(X)

        N = X.shape[1] if (self.n_components is None) else self.n_components
        self.column_names = [f"PCA{i+1}" for i in range(N)]
        return self

    def transform(self, X):
        check_is_fitted(self, 'reductor')
        return pd.DataFrame(self.reductor.transform(X), columns=self.column_names)

    def get_loadings(self, X):
        check_is_fitted(self, 'reductor')
        df_loadings = pd.DataFrame(self.reductor.components_, columns=X.columns,
                                     index=self.column_names)
        return df_loadings

    def explain_variance(self, n=None):
        expl_var = self.reductor.explained_variance_ratio_.cumsum()
        plt.plot(np.arange(1, len(expl_var)+1), expl_var)
        plt.xlabel('number of pca components')
        plt.ylabel('cumulative sum of explained variance')

        if n is not None:
            title = f"{expl_var[n-1] * 100:0.2f}% variance explained for {n} components"
            plt.title(title)
            plt.gca().axvline(n, color='r')

