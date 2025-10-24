import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TimeFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="date", use_year=True, use_month=True):
        self.datetime_col = datetime_col
        self.use_year = use_year
        self.use_month = use_month
        self.min_year = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        if self.use_year and self.datetime_col in X.columns:
            self.min_year = pd.to_datetime(X[self.datetime_col]).dt.year.min()
        self.non_datetime_features = [col for col in X.columns if col != self.datetime_col]
        return self

    def transform(self, X):
        Xt = X.copy()
        dates = pd.to_datetime(Xt[self.datetime_col])
        new_features = []

        if self.use_month:
            months = dates.dt.month
            Xt['month_sin'] = np.sin(2 * np.pi * months / 12)
            Xt['month_cos'] = np.cos(2 * np.pi * months / 12)
            new_features.extend(['month_sin', 'month_cos'])

        if self.use_year:
            years = dates.dt.year
            Xt['years_since_start'] = years - self.min_year
            new_features.append('years_since_start')

        Xt = Xt.drop(columns=[self.datetime_col])
        self.feature_names_out_ = self.non_datetime_features + new_features
        return Xt[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
