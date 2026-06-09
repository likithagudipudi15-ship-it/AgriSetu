import requests
import os
from flask import Flask, redirect, render_template, request, flash
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import pickle
from fertilizer import fertilizer_dic


import pandas as pd
from flask import Flask, render_template, request, flash

# ===================== Existing Disease Detection Setup =====================
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model_cnn = CNN.CNN(39)    
model_cnn.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model_cnn.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model_cnn(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# ===================== Crop Recommendation Setup =====================
try:
    crop_model = pickle.load(open('Model.pkl','rb'))
    sc = pickle.load(open('standscaler.pkl','rb'))
    ms = pickle.load(open('minmaxscaler.pkl','rb'))
except Exception as e:
    crop_model = None
    sc = None
    ms = None
    print("⚠️ Could not load crop recommendation model or scalers:", e)

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# ===================== Flask App =====================
app = Flask(__name__)
app.secret_key = "hackathon_secret"


@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title=title , desc=description , prevent=prevent , 
                               image_url=image_url , pred=pred , sname=supplement_name ,
                               simage=supplement_image_url , buy_link=supplement_buy_link)

@app.route('/market')
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

# ===================== NEW: Crop Recommendation Routes =====================
@app.route('/crop')
def crop_page():
    return render_template('crop.html')

@app.route('/crop/predict', methods=['POST'])
def crop_predict():
    if not crop_model or not sc or not ms:
        flash("Crop recommendation model not available. Please check server setup.", "danger")
        return render_template("crop.html")

    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        prediction = crop_model.predict(final_features)

        crop = crop_dict.get(prediction[0], "Unknown")
        result = f"{crop} is the best crop to be cultivated right now."
    except Exception as e:
        print("Error in prediction:", e)
        result = "Sorry, we could not determine the best crop. Please check input values."

    return render_template('crop.html', result=result)

# ===================== Fertilizer Recommendation (Optional) =====================

@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer_page():
    if request.method == 'POST':
        try:
            crop_name = request.form.get('crop', '').strip()
            N = float(request.form.get('nitrogen', '0') or 0)
            P = float(request.form.get('phosphorous', '0') or 0)
            K = float(request.form.get('potassium', '0') or 0)
        except Exception:
            flash("Please enter valid numeric values for N, P and K.", "warning")
            return render_template('fertilizer.html')

        df = pd.read_csv("fertilizer.csv")
        row = df[df['Crop'].str.lower() == crop_name.lower()]
        if row.empty:
            flash("Selected crop not found in fertilizer dataset.", "warning")
            return render_template('fertilizer.html')

        nr, pr, kr = float(row['N'].iloc[0]), float(row['P'].iloc[0]), float(row['K'].iloc[0])
        n, p, k = nr - N, pr - P, kr - K
        diffs = {abs(n): "N", abs(p): "P", abs(k): "K"}
        nutrient = diffs[max(diffs.keys())]

        if nutrient == "N":
            key = "NHigh" if n < 0 else "Nlow"
        elif nutrient == "P":
            key = "PHigh" if p < 0 else "Plow"
        else:
            key = "KHigh" if k < 0 else "Klow"

        response = fertilizer_dic.get(key, "No recommendation available.")

        return render_template('fertilizer.html',
                               response=response,   # send as plain string
                               crop_name=crop_name.title())
    return render_template('fertilizer.html')

@app.route('/weather')
def weather_page():
    return render_template('weather.html')
                           
# ----------------- Weather API Route -----------------

import requests

API_KEY = "bd1b0edf6d344feee41ecf87b5379e0f"  # Replace with your key


@app.route('/weather_api')
def weather_api():
    city = request.args.get('city')
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if lat and lon:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    elif city:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    else:
        return {"error": "City or coordinates required"}, 400

    try:
        resp = requests.get(url, timeout=5)
        data = resp.json()

        # Normalize cod to int
        data['cod'] = int(data.get('cod', 400))

        # Ensure optional fields exist
        data.setdefault('rain', {'1h': 0})
        data.setdefault('snow', {'1h': 0})
        data.setdefault('clouds', {'all': 0})
        data.setdefault('wind', {'speed': 0})
        data.setdefault('main', {'temp': 0, 'feels_like': 0, 'humidity': 0})
        data.setdefault('weather', [{'description': 'No data'}])
        data.setdefault('sys', {'country': ''})

        return data

    except Exception as e:
        print("Weather API fetch error:", e)
        return {"error": "Failed to fetch weather data", "cod": 500}, 500

crop_names = ["Rice", "Maize", "Jute", "Cotton", "Coconut", "Papaya", "Orange",
              "Apple", "Muskmelon", "Watermelon", "Grapes", "Mango", "Banana",
              "Pomegranate", "Lentil", "Blackgram", "Mungbean", "Mothbeans",
              "Pigeonpeas", "Kidneybeans", "Chickpea", "Coffee"]

models = {}
for crop in crop_names:
    with open(f'{crop}_model.pkl', 'rb') as f:
        models[crop] = pickle.load(f)

# Load dataset
df = pd.read_csv('Crop_Prices_Last_15_Days.csv')

@app.route('/price', methods=['GET', 'POST'])
def price():
    price = None
    selected_crop = None
    last_5_days_prices = None
    last_5_days_dates = None

    if request.method == 'POST':
        selected_crop = request.form.get('crop')
        if selected_crop in models:
            # Predict today's price
            last_row = df.iloc[-1][1:].values.reshape(1, -1)  # exclude Date
            model = models[selected_crop]
            price = round(model.predict(last_row)[0], 2)

            # Get last 5 days prices for selected crop
            last_5 = df[['Date', selected_crop]].tail(5)
            last_5_days_dates = last_5['Date'].tolist()
            last_5_days_prices = last_5[selected_crop].tolist()

    return render_template(
        'price.html',
        crops=crop_names,
        price=price,
        selected_crop=selected_crop,
        last_5_days_dates=last_5_days_dates,
        last_5_days_prices=last_5_days_prices
    )



if __name__ == '__main__':
    app.run(debug=True)

