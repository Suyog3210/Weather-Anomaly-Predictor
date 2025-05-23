# 🌦️ Weather Anomaly Detection and Extreme Event Prediction

![Weather](https://img.shields.io/badge/Project-Weather%20Analysis-blue)  
![License](https://img.shields.io/badge/License-Free-green)

---

## 📜 Project Overview

This project is a **comprehensive weather data analysis pipeline** built using **Python**, with a focus on:

- Data **cleaning** and **feature engineering** 🛠️
- **Anomaly detection** in weather parameters 🚨
- **Visual exploration** and plotting 📈
- **Random Forest-based prediction** for weather forecasting 🌳
- **Extreme weather event prediction** like cyclones and heatwaves 🌪🔥
- **Exporting** processed data and summaries for further insights 📂

The primary goal is to **detect anomalies** in historical weather data and **predict** future values and events using **machine learning** and **statistical techniques**.

---

## 📂 Project Structure

```
├── data/
│   ├── Weather_data.csv (input)
│   ├── processed_weather.csv (processed data)
│   ├── anomaly_summary.csv (anomaly count summary)
├── outputs/
│   ├── *.png (generated plots)
├── scripts/
│   ├── main.py (this script)
├── README.md
```

---

## 🔥 Key Features

- **Feature Engineering**  
  Rolling averages, lag features, and range calculations for enhanced predictive power.

- **Anomaly Detection**  
  Using **2-standard deviation** rule to flag unusual weather behavior across 11+ parameters.

- **Data Visualizations**  
  📊 Line plots, scatter plots, bar charts, boxplots, and heatmaps for deep insight.

- **Machine Learning Prediction**  
  🎯 Using **Random Forest Regressor** to predict important metrics like Temperature, Humidity, and Pressure.

- **Extreme Weather Detection**  
  🚨 Identifies conditions indicative of events like **Cyclones** and **Heatwaves** based on thresholds.

- **Automatic Exports**  
  📤 Save cleaned datasets, anomaly summaries, and all visual outputs.

---

## 📊 Visualizations

- Parameter anomaly detection plots
- Monthly anomaly distribution
- Correlation heatmap
- Box plots for feature distribution
- Actual vs Predicted plots for important features
- Extreme weather event prediction counts

_All plots are automatically saved as high-quality `.png` files!_

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas** 🐼
- **NumPy** 🔢
- **Matplotlib** 🎨
- **Seaborn** 🐳
- **Scikit-learn** 🤖

---

## ⚡ How to Run

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/weather-anomaly-detection.git
   cd weather-anomaly-detection
   ```

2. Install required packages:  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. Place your `Weather_data.csv` inside the `data/` directory.

4. Run the main script:  
   ```bash
   python scripts/main.py
   ```

5. Check the generated files inside `/outputs/`.

---

## 📢 Notes

- Ensure your date column is correctly formatted as `Date` in the CSV file.
- The anomaly threshold is set at **2× standard deviation** — you can tweak this for stricter/looser anomaly detection.
- The extreme weather event prediction can be extended by adding new conditions in the `extreme_weather_event_prediction()` function.

---

## 🤝 Credits

- **Developed by:** [Suyog3210]
- **Inspired by:** Real-world weather analytics needs in meteorology.
- **Special Thanks:** Open-source Python community!

---

## 📜 License

This project is free to use.
Feel free to fork, modify, and contribute!

---

## 📬 Connect

- ✉️ [Email](suyogpalve030@gmail.com)
- 🐙 [GitHub](https://github.com/Suyog3210)
  
