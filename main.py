import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score

# --------------------------------------
# Load and preprocess the data
# --------------------------------------
df = pd.read_csv('../data/Weather_data.csv', parse_dates=['Date'], dayfirst=True)
df.dropna(inplace=True)
df.sort_values('Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# --------------------------------------
# Feature Engineering
# --------------------------------------
df['Temp_Range'] = df['Temperature (Max)'] - df['Temperature (Min)']
df['Humidity_Range'] = df['Humidity(Max)'] - df['Humidity(Min)']
df['Pressure_Range'] = df['Pressure(Max)'] - df['Pressure(Min)']

rolling_cols = ['Temperature (Avg)', 'Humidity(Avg)', 'Pressure(Avg)', 'Wind Speed(Avg)', 'Dew Point(Avg)', 'Precipitation(Total)', 
                'Temperature (Max)', 'Temperature (Min)', 'Humidity(Max)', 'Humidity(Min)', 'Pressure(Max)', 'Pressure(Min)']
for col in rolling_cols:
    df[f'{col}_7d_avg'] = df[col].rolling(window=7).mean()

for col in ['Temperature (Avg)', 'Humidity(Avg)', 'Precipitation(Total)']:
    for lag in range(1, 4):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

df.dropna(inplace=True)
df.to_csv('../data/processed_weather.csv', index=False)

# --------------------------------------
# Anomaly Detection
# --------------------------------------
anomaly_cols = rolling_cols
for col in anomaly_cols:
    threshold = df[col].std() * 2
    df[f'{col}_Anomaly'] = abs(df[col] - df[col].mean()) > threshold

df['Anomaly_Flag'] = df[[f'{col}_Anomaly' for col in anomaly_cols]].any(axis=1).astype(int)
df['Month'] = df['Date'].dt.strftime('%B')

# --------------------------------------
# Save Plot Helper
# --------------------------------------
def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Saved: {filename}")

# --------------------------------------
# Visualization: Anomaly Plots
# --------------------------------------
def plot_anomalies(df):
    colors = {
        'Temperature(Avg)': '#0077b6',
        'Humidity(Avg)': '#00b4d8',
        'Wind Speed(Avg)': '#f94144',
        'Dew Point(Avg)': '#f3722c',
        'Precipitation(Total)': '#90be6d',
        'Pressure(Avg)': '#577590',
        'Temperature(Max)': '#ef476f',
        'Temperature(Min)': '#6c5b7b',
        'Humidity(Max)': '#f1faee',
        'Humidity(Min)': '#2a9d8f',
        'Pressure(Max)': '#264653',
        'Pressure(Min)': '#e9c46a'
    }

    # Loop through all columns and plot anomalies
    for col in anomaly_cols:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df['Date'], df[col], label=f'{col}', color=colors.get(col, '#000000'))
        ax.scatter(df['Date'][df[f'{col}_Anomaly']], df[col][df[f'{col}_Anomaly']], color='red', label='Anomaly', s=20)
        ax.set_title(f'{col} Anomalies')
        ax.set_xlabel('Date')
        ax.set_ylabel(col)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        save_plot(fig, f'{col}_anomalies.png')

        # Scatter plot for anomalies
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.scatter(df['Date'], df[col], label=f'{col}', color=colors.get(col, '#000000'), s=10)
        ax.scatter(df['Date'][df[f'{col}_Anomaly']], df[col][df[f'{col}_Anomaly']], color='red', label='Anomaly', s=30)
        ax.set_title(f'{col} Scatter Plot with Anomalies')
        ax.set_xlabel('Date')
        ax.set_ylabel(col)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        save_plot(fig, f'{col}_scatter_anomalies.png')

    # Anomaly count bar chart
    anomaly_counts = [df[f'{col}_Anomaly'].sum() for col in anomaly_cols]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(anomaly_cols, anomaly_counts, color=list(colors.values()))
    ax.set_title('Total Anomalies per Weather Parameter')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    save_plot(fig, 'anomaly_bar_counts.png')

    # Monthly Anomaly Distribution
    monthly_counts = df.groupby('Month', group_keys=False)['Anomaly_Flag'].sum()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_counts = monthly_counts.reindex(month_order)
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_counts.plot(kind='bar', color='orchid', ax=ax)
    ax.set_title('Monthly Anomaly Distribution')
    ax.set_ylabel('Number of Anomalies')
    save_plot(fig, 'monthly_anomaly_distribution.png')

    # Correlation Heatmap for all columns (not just Avg)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df[rolling_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Between Weather Parameters')
    save_plot(fig, 'correlation_heatmap_full.png')

    # Boxplot for all parameters
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=df[rolling_cols], palette='Set3', ax=ax)
    ax.set_title('Box Plot of Weather Parameters')
    ax.set_ylabel('Values')
    save_plot(fig, 'boxplot_weather_full.png')

# --------------------------------------
# Random Forest Prediction Visualization
# --------------------------------------
def prediction_analysis(df, target, color):
    feature_cols = [col for col in df.columns if 'lag' in col or '7d_avg' in col]
    train = df[df['Date'] < '2024-10-01']
    test = df[df['Date'] >= '2024-10-01'].copy()

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train[feature_cols], train[target])
    test['Predicted'] = rf.predict(test[feature_cols])

    mae = mean_absolute_error(test[target], test['Predicted'])
    rmse = np.sqrt(mean_squared_error(test[target], test['Predicted']))
    r2 = r2_score(test[target], test['Predicted'])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test['Date'], test[target], label='Actual', color='black')
    ax.plot(test['Date'], test['Predicted'], label='Predicted', color=color)
    ax.set_title(f'{target} - Random Forest Prediction\nMAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}')
    ax.set_xlabel('Date')
    ax.set_ylabel(target)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    save_plot(fig, f'{target}_rf_prediction.png')

# --------------------------------------
# Run All Visualizations and Predictions
# --------------------------------------
plot_anomalies(df)

prediction_targets = {
    'Temperature (Avg)': '#ff6f61',
    'Humidity(Avg)': '#1982c4',
    'Wind Speed(Avg)': '#8e44ad',
    'Precipitation(Total)': '#43aa8b',
    'Pressure(Avg)': '#ffbe0b',
    'Dew Point(Avg)': '#d00000',
    'Temperature (Max)': '#ff6347',
    'Temperature (Min)': '#32cd32'
}

for target, color in prediction_targets.items():
    prediction_analysis(df, target, color)

# Export anomaly counts to CSV
anomaly_summary = pd.DataFrame({
    'Parameter': anomaly_cols,
    'Anomaly_Count': [df[f'{col}_Anomaly'].sum() for col in anomaly_cols]
})
anomaly_summary.to_csv('anomaly_summary.csv', index=False)
print("Anomaly summary exported.")

# --------------------------------------
# Extreme Weather Prediction
# --------------------------------------
def extreme_weather_event_prediction(df):
    conditions = {
        '🌪 Cyclone': {
            'Temperature (Max)': 30,
            'Temperature (Avg)': 26,
            'Temperature (Min)': 22,
            'Wind Speed(Avg)': 100,
            'Pressure(Min)': 28.64,
            'Precipitation(Total)': 100
        },
        '🔥 Heatwave': {
            'Temperature (Max)': 45,
            'Temperature (Avg)': 38,
            'Temperature (Min)': 28,
            'Wind Speed(Avg)': 10,
            'Pressure(Min)': 29.68,
            'Precipitation(Total)': 0
        },
        # Add other events with the same structure...
    }

    event_predictions = []
    for _, row in df.iterrows():
        predicted_event = None
        for event, condition in conditions.items():
            if all(row[param] >= value for param, value in condition.items()):
                predicted_event = event
                break
        event_predictions.append(predicted_event)

    df['Predicted_Event'] = event_predictions

    # Plotting predictions
    fig, ax = plt.subplots(figsize=(14, 6))
    event_counts = df['Predicted_Event'].value_counts()
    ax.bar(event_counts.index, event_counts.values, color='royalblue')
    ax.set_title('Extreme Weather Event Predictions')
    ax.set_xlabel('Event')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    save_plot(fig, 'extreme_weather_predictions.png')

# Run Extreme Weather Prediction
extreme_weather_event_prediction(df)
