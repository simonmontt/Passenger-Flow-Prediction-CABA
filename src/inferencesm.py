from datetime import datetime, timedelta
import hopsworks
import pandas as pd
import numpy as np
import hsfs
import src.config as config
from src.feature_store_api import get_feature_store


def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches the predictions from the model and rounds them.
    """
    predictions = model.predict(features)
    predictions = predictions.round(0)

    results = pd.DataFrame(predictions, columns=[f'total_pax_next_{i+1}_hour' for i in range(3)])
    results['station'] = features['station'].values
    results['line'] = features['line'].values

    return results

def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """
    Fetches the batch of features used by the ML system at `current_date`.

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features.

    Returns:
        pd.DataFrame: DataFrame containing 4 columns:
        `hour_of_entry`,
        `total_pax`,
        `line`,
        `station`.
    """
    n_features = config.N_FEATURES

    feature_store = get_feature_store()

    # Define the period to fetch data for the model
    fetch_data_to = pd.to_datetime(current_date - timedelta(hours=1), utc=True)
    fetch_data_from = pd.to_datetime((current_date - timedelta(days=14)) - timedelta(hours=2), utc=True)# - timedelta(hours=2)) )
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')

    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time=pd.to_datetime(fetch_data_from, utc=True),
        end_time=pd.to_datetime(fetch_data_to, utc=True)
    )

    # Convert 'hour_of_entry' to UTC-aware datetime
    ts_data['hour_of_entry'] = pd.to_datetime(ts_data['hour_of_entry'], utc=True)

    # Filter data to the time period we are interested in
    ts_data = ts_data[ts_data.hour_of_entry.between(fetch_data_from, fetch_data_to)]

    # Validate the presence of required data for all stations and lines
    station_line_ids = ts_data[['station', 'line']].drop_duplicates()
    expected_length = n_features * len(station_line_ids)
    print(station_line_ids.columns)
    if len(ts_data) != expected_length:
        raise ValueError(f"Time-series data is incomplete. Expected {expected_length} rows, but got {len(ts_data)}. Please ensure the feature pipeline is running properly.")

    # Sort data by station, line, and time
    ts_data.sort_values(by=['station', 'line', 'hour_of_entry'], inplace=True)

    # Transpose time-series data as a feature vector for each station-line combination
    x = np.ndarray(shape=(len(station_line_ids), n_features), dtype=np.float32)
    for i, (station, line) in enumerate(station_line_ids.values):
        ts_data_i = ts_data[(ts_data['station'] == station) & (ts_data['line'] == line)]
        ts_data_i = ts_data_i.sort_values(by=['hour_of_entry'])
        x[i, :] = ts_data_i['total_pax'].values

    # Create DataFrame of features
    features = pd.DataFrame(
        x,
        columns=[f'total_pax_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )

    features['hour_of_entry'] = pd.to_datetime(current_date, utc=True)
    features['station'] = station_line_ids['station'].values
    features['line'] = station_line_ids['line'].values
    features.sort_values(by=['station', 'line'], inplace=True)

    return features

def load_model_from_registry():
    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / 'model222324.pkl')
    
    return model

def load_predictions_from_store(from_hour_of_entry: datetime, to_hour_of_entry: datetime) -> pd.DataFrame:
    """
    Fetches model predictions for all stations and lines for the time period
    between `from_hour_of_entry` and `to_hour_of_entry`.

    Args:
        from_hour_of_entry (datetime): Min datetime for predictions.
        to_hour_of_entry (datetime): Max datetime for predictions.

    Returns:
        pd.DataFrame: DataFrame containing predictions for each station and line.
    """
    feature_store = get_feature_store()

    prediction_fg = feature_store.get_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTIONS,
        version=1,
    )

    try:
        # Create the feature view if it doesn't already exist
        feature_store.create_feature_view(
            name=config.FEATURE_VIEW_MODEL_PREDICTIONS,
            version=1,
            query=prediction_fg.select_all()
        )
    except Exception as e:
        print(f"Feature view {config.FEATURE_VIEW_MODEL_PREDICTIONS} already exists. Skipping creation. Error: {e}")

    predictions_fv = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_MODEL_PREDICTIONS,
        version=1
    )

    print(f'Fetching predictions between {from_hour_of_entry} and {to_hour_of_entry}')
    predictions = predictions_fv.get_batch_data(
        #start_time=from_hour_of_entry - timedelta(days=1),
        #end_time=to_hour_of_entry + timedelta(days=1)
    )

    # Ensure datetimes are UTC-aware
    predictions['hour_of_entry'] = pd.to_datetime(predictions['hour_of_entry'], utc=True)
    #from_hour_of_entry = pd.to_datetime(from_hour_of_entry, utc=True)
    #to_hour_of_entry = pd.to_datetime(to_hour_of_entry, utc=True)

    #predictions = predictions[predictions.hour_of_entry.between(
    #    from_hour_of_entry, to_hour_of_entry
    #)]

    # Sort predictions by 'hour_of_entry', 'station', and 'line'
    predictions.sort_values(by=['hour_of_entry', 'station', 'line'], inplace=True)

    return predictions
