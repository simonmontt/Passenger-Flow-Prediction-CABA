import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from src.inferencesm import load_batch_of_features_from_store, load_predictions_from_store

# Loading the image using Streamlit's st.image
st.image("Logos.png", width=200, use_column_width=False)

# Title for the app
st.title("Buenos Aires Subway passenger flow")

# Sidebar - Progress and Loading Indicators
with st.sidebar:
    st.write("### Loading Status")

# Function for loading batch of features with Streamlit caching
@st.cache_data
def cached_load_batch_of_features() -> pd.DataFrame:
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    return load_batch_of_features_from_store(current_time)

# Function for loading predictions with Streamlit caching
@st.cache_data
def cached_load_predictions() -> pd.DataFrame:
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    from_hour_of_entry = current_time
    to_hour_of_entry = from_hour_of_entry + timedelta(hours=2)
    return load_predictions_from_store(from_hour_of_entry, to_hour_of_entry)

# Function for loading historical data for comparison (previous year's data)
@st.cache_data
def load_historical_data(year: int, station: str, line: str) -> pd.DataFrame:
    # Load data from your historical data source (this is just an example)
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    previous_year_time = current_time - timedelta(days=364)# + pd.Timedelta(hours=3)
    historical_data = load_batch_of_features_from_store(previous_year_time)  # Replace with actual historical data loading function
    return historical_data[(historical_data['station'] == station) & (historical_data['line'] == line)]

# Function to plot total passengers and compare with historical data
def plot_total_pax_with_comparison(features_df: pd.DataFrame, predictions_df: pd.DataFrame, line: str, station: str, current_time: datetime):
    filtered_features = features_df[(features_df['line'] == line) & (features_df['station'] == station)]
    filtered_predictions = predictions_df[(predictions_df['line'] == line) & (predictions_df['station'] == station)]

    if filtered_features.empty or filtered_predictions.empty:
        st.error("No data available for the selected line and station.")
        return

    # Get historical data for the same date last year
    last_year_data = load_historical_data(current_time.year - 1, station, line)
    
    if last_year_data.empty:
        st.error("No historical data available for comparison.")
        return
    
    # Get the last 24 hours of passenger data
    total_pax_previous_cols = filtered_features.filter(like='total_pax_previous').columns[-24:]
    total_pax_previous = filtered_features[total_pax_previous_cols].iloc[0].values
    
    # Get the next 3 hours of passenger predictions
    total_pax_next = filtered_predictions.filter(like='total_pax_next').iloc[0].values[:3]
    
    # Historical data for comparison (matching the predicted hours)
    historical_pax_next = last_year_data.filter(like='total_pax_previous').iloc[0].values[-3:]

    # Create time series for the last 24 hours and next 3 hours
    time_series_previous = pd.date_range(end=current_time, periods=24, freq='H')
    time_series_next = pd.date_range(start=current_time + pd.Timedelta(hours=1), periods=3, freq='H')
    
    # Combine previous and next time series
    time_series = time_series_previous.append(time_series_next)
    pax_series = np.concatenate([total_pax_previous, total_pax_next])

    # Calculate PMAE between predicted and actual values from last year
    #actual_safe = np.where(historical_pax_next == 0, 1e-9, historical_pax_next)  # Avoid division by zero
    #pmae = np.mean(np.abs((total_pax_next - historical_pax_next) / actual_safe)) * 100
    mae = mean_absolute_error(historical_pax_next, total_pax_next)
    
    # Create Plotly graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_series[:24], y=total_pax_previous, mode='lines+markers', name='Actual Total Pax (Last 24 hours)', line=dict(color='blue'),  hovertemplate='Date: %{x}<br>Total Pax: %{y}<extra></extra>'))
    fig.add_trace(go.Scatter(x=time_series[24:], y=total_pax_next, mode='lines+markers', name='Predicted Total Pax (Next 3 hours)', line=dict(color='orange'), hovertemplate='Date: %{x}<br>Predicted Pax: %{y}<extra></extra>'))
    fig.add_trace(go.Scatter(x=time_series[24:], y=historical_pax_next, mode='lines+markers', name='Actual Pax Last Year (Next 3 hours)', line=dict(color='green', dash='dash'), hovertemplate='Date: %{x}<br>Last Year Pax: %{y}<extra></extra>'))

    fig.update_layout(
        title=f'Total Passenger Flow for Line {line}, Station {station}/n <br>YoY Difference: {mae:.2f}',
        xaxis_title='Date and Time',
        yaxis_title='Total Passengers',
        template='plotly_dark',  # Adjust background color to fit the UI
        plot_bgcolor='rgba(43, 43, 43, 1)',  # Background color to match Streamlit theme
        paper_bgcolor='rgba(30, 30, 30, 1)',
        xaxis=dict(tickangle=-45),
        hovermode='x unified'
    )

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)

# Sidebar progress messages
with st.sidebar:
    with st.spinner('Loading features...'):
        features = cached_load_batch_of_features()
    with st.spinner('Loading predictions...'):
        predictions = cached_load_predictions()

    st.success('Data Loaded Successfully!')

# Encoding labels for line and station
line_label_encoder = LabelEncoder()
station_label_encoder = LabelEncoder()

features['line_encoded'] = line_label_encoder.fit_transform(features['line'])
features['station_encoded'] = station_label_encoder.fit_transform(features['station'])
predictions['line'] = line_label_encoder.inverse_transform(predictions['line'])
predictions['station'] = station_label_encoder.inverse_transform(predictions['station'])

print("PRED")
print(predictions, predictions.shape)
print("DREP")
# Get unique lines for dropdowns
unique_lines = sorted(features['line'].unique())

# Dropdown for selecting Line
selected_line_encoded = st.selectbox("Select Line", unique_lines)

# Get stations corresponding to the selected line
filtered_stations = features[features['line'] == selected_line_encoded]['station'].unique()

# Dropdown for selecting Station
selected_station_encoded = st.selectbox("Select Station", filtered_stations)

if st.button("Generate Plot"):
    plot_total_pax_with_comparison(features, predictions, selected_line_encoded, selected_station_encoded, datetime.now())
