import pandas as pd

def load_rainfall_data(filepath):
    """Loads and preprocesses India's rainfall dataset."""
    df = pd.read_csv(filepath)

    # Keep relevant columns
    df = df[['YEAR', 'JUN', 'JUL', 'AUG', 'SEP', 'JUN-SEP']]

    # Rename columns for consistency
    df.rename(columns={'YEAR': 'Year', 'JUN-SEP': 'Total_Monsoon_Rainfall'}, inplace=True)

    return df


def load_disaster_data(filepath):
    """Loads and preprocesses disaster data for analytics and prediction."""
    df = pd.read_excel(filepath, engine='openpyxl')

    # Keep only necessary columns for analytics
    df = df[['Start Year', 'Disaster Type', 'Disaster Subtype', 'Magnitude',
             'Total Deaths', 'No. Injured', 'No. Affected', 'Total Damage (\'000 US$)']]

    # Rename columns for consistency
    df.rename(columns={'Start Year': 'Year', 'Total Damage (\'000 US$)': 'Total Damage'}, inplace=True)

    return df
