import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.multioutput import MultiOutputRegressor
import hopsworks
import src.config as cfg

import lightgbm as lgb
import holidays

def average_total_pax_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides from
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    """
    X['average_rides_last_4_weeks'] = 0.25*(
        X[f'total_pax_previous_{7*24}_hour']
        #X[f'total_pax_previous_{2*7*24}_hour'] + \
        #X[f'total_pax_previous_{3*7*24}_hour'] + \
        #X[f'total_pax_previous_{4*7*24}_hour']
    )
    return X

def add_peak_hour_feature(X: pd.DataFrame) -> pd.DataFrame:
    X['is_peak_hour'] = X['hour'].apply(lambda x: 1 if 7 <= x <= 9 or 17 <= x <= 19 else 0)
    return X

def add_rolling_features(X: pd.DataFrame) -> pd.DataFrame:
    X['rolling_mean_24*7_hours'] = X[f'total_pax_previous_{7*24}_hour'].mean()
    X['rolling_std_24*7_hours'] = X[f'total_pax_previous_{7*24}_hour'].std()
    #X['ema_24_hours'] = X[f'total_pax_previous_{7*24}_hour'].ewm(span=24).mean()
    return X


arg_holidays = holidays.CountryHoliday('AR')
def add_holiday_flag(X: pd.DataFrame) -> pd.DataFrame:
    X['is_holiday'] = X['hour_of_entry'].apply(lambda x: 1 if x in arg_holidays else 0)
    return X

class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds 2 columns
    - hour
    - day_of_week
    and removes the `hour_of_entry` datetime column.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        # Generate numeric columns from datetime
        X_["hour"] = X_['hour_of_entry'].dt.hour
        X_["day_of_week"] = X_['hour_of_entry'].dt.dayofweek
        X_['week_of_year'] = X_['hour_of_entry'].dt.isocalendar().week
        
        return X_.drop(columns=['hour_of_entry'])

def get_pipeline(**hyperparams) -> Pipeline:
    
    #add_feature_average_rides_last_4_weeks = FunctionTransformer(
    #    average_total_pax_last_4_weeks, validate=False)

    add_temporal_features = TemporalFeaturesEngineer()
    
    add_peak_hour_transform = FunctionTransformer(add_peak_hour_feature, validate=False)
    add_rolling_feature_transform = FunctionTransformer(add_rolling_features, validate=False)
    add_holiday_flag_transform = FunctionTransformer(add_holiday_flag, validate=False)

    return make_pipeline(
        #add_feature_average_rides_last_4_weeks,
        add_holiday_flag_transform,
        add_temporal_features,
        add_peak_hour_transform,
        add_rolling_feature_transform,
        MultiOutputRegressor(lgb.LGBMRegressor(**hyperparams, force_col_wise=True))
    )