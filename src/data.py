import pandas as pd

def fill_missing_datetimes(df):
    # Convert 'fecha_de_inicio' to datetime if not already
    df['hour_of_entry'] = pd.to_datetime(df['hour_of_entry'])

    
    # Initialize an empty DataFrame to store the results
    filled_df = pd.DataFrame()
    
    # Group by 'linea' and 'station' to ensure separation between groups
    for (line, station), group in df.groupby(['line', 'station']):
        # Sort by 'fecha_de_inicio' just in case
        group = group.sort_values('hour_of_entry')
        
        # Create a complete hourly range for this group
        full_range = pd.date_range(start=group['hour_of_entry'].min(), 
                                   end=group['hour_of_entry'].max(), 
                                   freq='H')
        
        # Set 'fecha_de_inicio' as the index to reindex with full_range
        group = group.set_index('hour_of_entry')
        
        # Reindex to fill missing datetimes and set 'total_pax' to 0 where missing
        group = group.reindex(full_range, fill_value=0).reset_index()
        
        # Rename the index back to 'fecha_de_inicio'
        group = group.rename(columns={'index': 'hour_of_entry'})
        
        # Restore 'linea' and 'station' columns with the correct values
        group['line'] = line
        group['station'] = station
        
        # Append the filled group to the final DataFrame
        filled_df = pd.concat([filled_df, group], ignore_index=True)
    
    return filled_df


def load_and_concatenate(file1, file2):
    # Load both files
    df1 = pd.read_csv(file1, encoding='ISO-8859-1')
    df2 = pd.read_csv(file2, encoding='ISO-8859-1')
    
    # Drop the 'Unnamed: 0' column
    df1.drop(columns="Unnamed: 0", inplace=True)
    df2.drop(columns="Unnamed: 0", inplace=True)
    
    # Concatenate both DataFrames
    concatenated_df = pd.concat([df1, df2], ignore_index=True)
    
    concatenated_df.rename(columns={'fecha_de_inicio': 'hour_of_entry', 'linea': 'line'}, inplace=True)
    
    return concatenated_df
