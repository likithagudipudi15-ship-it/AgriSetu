import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from datetime import datetime

# Load dataset
df = pd.read_csv('Crop_Prices_Last_15_Days.csv')

# Prepare training data
# Using previous 14 days to predict the next day
X = []
y_dict = {crop: [] for crop in df.columns if crop != 'Date'}

# Build sliding window of size 14
for i in range(len(df) - 1):
    X.append(df.iloc[i][1:].values)  # exclude Date
    for crop in y_dict:
        y_dict[crop].append(df.iloc[i + 1][crop])

X = pd.DataFrame(X, columns=[col for col in df.columns if col != 'Date'])

# Train a Random Forest model for each crop
models = {}
for crop in y_dict:
    y = y_dict[crop]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    models[crop] = model
    # Save model to pickle
    with open(f'{crop}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

print("Models trained and saved as pickle files.")

# Function to predict today's price given crop name
def predict_price(crop_name):
    if crop_name not in models:
        return f"Crop '{crop_name}' not found in dataset."
    # Use last available day's prices as input
    last_row = df.iloc[-1][1:].values.reshape(1, -1)  # exclude Date
    model = models[crop_name]
    predicted_price = model.predict(last_row)[0]
    return round(predicted_price, 2)

# Example usage:
crop_input = input("Enter the crop name to predict today's price: ")
price = predict_price(crop_input)
print(f"Predicted today's price for {crop_input}: ₹{price}")
