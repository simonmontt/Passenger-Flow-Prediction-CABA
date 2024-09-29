from datetime import datetime, timedelta
import pandas as pd
import src.config as cfg
from src.feature_store_api import get_feature_store

def load_predictions_and_actual_values_from_store(
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    """
    Fetches model predictions and actual values from the Feature Store
    within the specified date range and returns two DataFrames.

    Args:
        from_date (datetime): Start date to fetch predictions and actual values.
        to_date (datetime): End date to fetch predictions and actual values.

    Returns:
        tuple: Two DataFrames - one for predicted demand and another for actual rides.
    """
    # Get the current UTC datetime, floored to the nearest hour
    current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')

    # Set the fetch date range for data
    fetch_data_from = pd.Timestamp('2023-01-01 00:00:00+0000', tz='UTC')
    fetch_data_to = pd.to_datetime(current_date - timedelta(hours=1), utc=True)

    # Fetch predictions from the feature store
    feature_store_1 = get_feature_store()
    predictions_fg = feature_store_1.get_feature_view(name=cfg.FEATURE_VIEW_MODEL_PREDICTIONS)
    ts_data_1 = predictions_fg.get_batch_data(
        start_time=pd.to_datetime(fetch_data_from, utc=True),
        end_time=pd.to_datetime(fetch_data_to, utc=True)
    )

    # Fetch actual values from the feature store
    feature_store_2 = get_feature_store()
    actuals_fg = feature_store_2.get_feature_view(name=cfg.FEATURE_VIEW_NAME)
    ts_data_2 = actuals_fg.get_batch_data(
        start_time=pd.to_datetime(fetch_data_from, utc=True),
        end_time=pd.to_datetime(fetch_data_to, utc=True)
    )

    return ts_data_1, ts_data_2
