
import pandas as pd
import joblib

# Load the trained model
bundle = joblib.load("artifacts/online_retail_sales_model.pkl")
model = bundle["model"]
feature_columns = bundle["feature_columns"]

# Sample data for predicting daily revenue
# Replace these example values with actual historical sales values
sales_data = {
    "DayOfWeek": 0,       # Monday = 0, Sunday = 6
    "DayOfMonth": 12,
    "Month": 10,
    "IsWeekend": 0,       # 0 = weekday, 1 = weekend
    "Lag_1": 1250.00,     # Revenue 1 day ago
    "Lag_7": 1100.00,     # Revenue 7 days ago
    "Lag_14": 980.00,     # Revenue 14 days ago
    "RollingMean_7": 1150.00,   # Average revenue over previous 7 days
    "RollingMean_14": 1080.00   # Average revenue over previous 14 days
}

# Convert dictionary to DataFrame
sales_df = pd.DataFrame([sales_data])

# Ensure features are in the same order used during training
sales_df = sales_df[feature_columns]

# Make prediction
prediction = model.predict(sales_df)

# Display result
print("Daily Sales Prediction")
print("----------------------")
print(f"Predicted revenue: {prediction[0]:.2f}")
