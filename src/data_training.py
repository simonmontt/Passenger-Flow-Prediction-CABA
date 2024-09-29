from tqdm import tqdm
import pandas as pd
import numpy as np

def get_cutoff_indices(
    data: pd.DataFrame,
    n_features: int,
    step_size: int,
    output_seq_len: int
) -> list:
    stop_position = len(data) - 1
    
    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + output_seq_len
    indices = []
    
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_mid_idx, subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size

    return indices

def transform_ts_data_into_features_and_target_old(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int,
    output_seq_len: int
) -> pd.DataFrame:
    assert set(ts_data.columns) == {'station', 'hour_of_entry', 'total_pax', 'line'}

    location_ids = ts_data['station'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        ts_data_one_location = ts_data.loc[
            ts_data.station == location_id, 
            ['hour_of_entry', 'total_pax']
        ].sort_values(by='hour_of_entry')

        indices = get_cutoff_indices(
            ts_data_one_location,
            input_seq_len,
            step_size,
            output_seq_len
        )

        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples, output_seq_len), dtype=np.float32)
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['total_pax'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['total_pax'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['hour_of_entry'])

        # Convert numpy arrays to pandas DataFrames
        features_one_location = pd.DataFrame(
            x,
            columns=[f'total_pax_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['hour_of_entry'] = pickup_hours
        features_one_location['station'] = location_id

        targets_one_location = pd.DataFrame(
            y, 
            columns=[f'total_pax_next_{i+1}_hour' for i in range(output_seq_len)]
        )

        # Concatenate results
        features = pd.concat([features, features_one_location], ignore_index=True)
        targets = pd.concat([targets, targets_one_location], ignore_index=True)

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets

def transform_ts_data_into_features_and_target(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int,
    output_seq_len: int
) -> pd.DataFrame:
    assert set(ts_data.columns) == {'station', 'hour_of_entry', 'total_pax', 'line'}

    # Unique combinations of station and line
    station_line_combinations = ts_data[['station', 'line']].drop_duplicates()

    features = pd.DataFrame()
    targets = pd.DataFrame()

    # Iterate over each station-line combination
    for _, row in tqdm(station_line_combinations.iterrows(), total=len(station_line_combinations)):
        location_id = row['station']
        line_id = row['line']

        # Filter data by both station and line
        ts_data_one_location = ts_data.loc[
            (ts_data.station == location_id) & (ts_data.line == line_id),
            ['hour_of_entry', 'total_pax', 'line']
        ].sort_values(by='hour_of_entry')

        indices = get_cutoff_indices(
            ts_data_one_location,
            input_seq_len,
            step_size,
            output_seq_len
        )

        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_seq_len), dtype=np.float32)
        y = np.ndarray(shape=(n_examples, output_seq_len), dtype=np.float32)
        pickup_hours = []

        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['total_pax'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['total_pax'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['hour_of_entry'])

        # Convert numpy arrays to pandas DataFrames
        features_one_location = pd.DataFrame(
            x,
            columns=[f'total_pax_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
        )
        features_one_location['hour_of_entry'] = pickup_hours
        features_one_location['station'] = location_id
        features_one_location['line'] = line_id  # Add line information here

        targets_one_location = pd.DataFrame(
            y, 
            columns=[f'total_pax_next_{i+1}_hour' for i in range(output_seq_len)]
        )

        # Concatenate results
        features = pd.concat([features, features_one_location], ignore_index=True)
        targets = pd.concat([targets, targets_one_location], ignore_index=True)

    features.reset_index(inplace=True, drop=True)
    targets.reset_index(inplace=True, drop=True)

    return features, targets

def transform_ts_data_into_dataset_comparable_with_predictions(
    ts_data: pd.DataFrame,
    input_seq_len: int,
    step_size: int,
    output_seq_len: int
) -> pd.DataFrame:
    """
    Transforms a time-series dataset into a format comparable with predictions,
    slicing and transposing data from time-series format into a (features, target)
    format for supervised ML models.

    Args:
        ts_data (pd.DataFrame): Time-series data with columns `hour_of_entry`, `rides`, `station`, `line`
        input_seq_len (int): Number of hours to use as input features (look-back period)
        step_size (int): Steps between sequences
        output_seq_len (int): Number of hours to predict (look-forward period)

    Returns:
        pd.DataFrame: A DataFrame containing `station`, `line`, `hour_of_entry`, and the target columns 
                      (e.g., `real_rides_next_{i+1}_hour` for each prediction horizon).
    """
    assert set(ts_data.columns) == {'hour_of_entry', 'station', 'line'}

    lines = ts_data['line'].unique()
    targets = pd.DataFrame()

    for line in lines:
        # Filter data by line
        ts_data_line = ts_data[ts_data['line'] == line]
        stations = ts_data_line['station'].unique()

        for station in stations:
            # Filter data for this station
            ts_data_station = ts_data_line.loc[
                ts_data_line.station == station,
                ['hour_of_entry', 'rides']
            ].sort_values(by=['hour_of_entry'])

            # Pre-compute cutoff indices to split dataframe rows
            indices = get_cutoff_indices(
                ts_data_station,
                input_seq_len,
                step_size,
                output_seq_len
            )

            # Slice and transpose data into numpy arrays for targets
            n_examples = len(indices)
            y = np.ndarray(shape=(n_examples, output_seq_len), dtype=np.float32)
            entry_hours = []

            for i, idx in enumerate(indices):
                y[i] = ts_data_station.iloc[idx[1]:idx[2]]['rides'].values
                entry_hours.append(ts_data_station.iloc[idx[1]]['hour_of_entry'])

            # Convert numpy array to pandas DataFrame
            targets_one_station = pd.DataFrame(y, columns=[f'real_rides_next_{i+1}_hour' for i in range(output_seq_len)])
            targets_one_station['hour_of_entry'] = entry_hours
            targets_one_station['station'] = station
            targets_one_station['line'] = line

            # Concatenate results
            targets = pd.concat([targets, targets_one_station])

    targets.reset_index(inplace=True, drop=True)

    return targets
