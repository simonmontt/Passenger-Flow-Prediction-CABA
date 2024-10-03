import os
import streamlit as st
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# load key-value pairs from .env file located in the parent directory
load_dotenv(PARENT_DIR / '.env')


#try:
#HOPSWORKS_PROJECT_NAME = 'mlops_rm'
#HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
#    HOPSWORKS_API_KEY = st.secrets["hopsworks"]["HOPSWORKS_API_KEY"]
#except:
#    raise Exception('Create an .env file on the project root with the HOPSWORKS_API_KEY')


# If not found in environment variables, use Streamlit secrets (for Streamlit Cloud)
#if not HOPSWORKS_API_KEY:
try:
        HOPSWORKS_API_KEY = st.secrets["hopsworks"]["HOPSWORKS_API_KEY"]
        HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except KeyError:
        raise Exception('HOPSWORKS_API_KEY not found in environment variables or Streamlit secrets.')

FEATURE_GROUP_NAME = 'ts_stations_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = 'ts_stations_hourly_feature_view'
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "Passenger_flow_predictor_next_full_day"
MODEL_VERSION = 1

# added for monitoring purposes
FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group_'
FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view_'
FEATURE_VIEW_MONITORING = 'predictions_vs_actuals_for_monitoring_feature_view'

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 14

# maximum Mean Absolute Error we allow our production model to have
#MAX_MAE = 50.0


import os
import streamlit as st

# Try to get the HOPSWORKS_API_KEY from environment variables (for GitHub Actions or local)
HOPSWORKS_PROJECT_NAME = 'mlops_rm'
HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')

# If not found in environment variables, use Streamlit secrets (for Streamlit Cloud)
if not HOPSWORKS_API_KEY:
    try:
        HOPSWORKS_API_KEY = st.secrets["hopsworks"]["HOPSWORKS_API_KEY"]
    except KeyError:
        raise Exception('HOPSWORKS_API_KEY not found in environment variables or Streamlit secrets.')
