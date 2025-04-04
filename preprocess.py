import pandas as pd

def merge_datasets(rainfall_df, disaster_df):
    """Merges rainfall and disaster data on Year and aggregates disaster events."""

    # Count disaster events per year
    disaster_count = disaster_df.groupby('Year').size().reset_index(name='Disaster_Count')

    # Merge with rainfall data
    merged = pd.merge(rainfall_df, disaster_count, on='Year', how='left')

    # Fill missing disaster counts with 0
    merged['Disaster_Count'] = merged['Disaster_Count'].fillna(0)

    # Create a binary column indicating if a disaster occurred
    merged['Disaster_Occurred'] = merged['Disaster_Count'].apply(lambda x: 1 if x > 0 else 0)

    return merged
