# app.py
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Load the scaler (assuming you saved it during training)
scaler = StandardScaler()
# If you saved the scaler, load it here. Example:
import joblib
scaler = joblib.load('scaler.save')

# Load the original dataset to get category mappings
df = pd.read_csv('Clean_Dataset.csv')
df = df.drop('Unnamed: 0', axis=1)  # Drop unnecessary column

# Encode categorical variables in the dataset (same as during training)
df_copy = df.copy()
for col in df_copy.select_dtypes(include=['object']).columns:
    df_copy[col] = df_copy[col].astype('category').cat.codes

# Create a dictionary to store category mappings
category_mappings = {}
for col in df.select_dtypes(include=['object']).columns:
    category_mappings[col] = dict(zip(df[col].astype('category').cat.categories, df_copy[col].astype('category').cat.codes))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = {
            'airline': request.form['airline'],
            'flight': request.form['flight'],
            'source_city': request.form['source_city'],
            'departure_time': request.form['departure_time'],
            'stops': request.form['stops'],
            'arrival_time': request.form['arrival_time'],
            'destination_city': request.form['destination_city'],
            'class': request.form['class'],
            'duration': float(request.form['duration']),
            'days_left': float(request.form['days_left']),
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Encode categorical variables using the same mappings as during training
        for col in input_df.select_dtypes(include=['object']).columns:
            input_df[col] = input_df[col].map(category_mappings[col])

        # Preprocess the input data (same as during training)
        input_scaled = scaler.transform(input_df)  # Scale the input

        # Make prediction
        prediction_log = model.predict(input_scaled)
        prediction = np.expm1(prediction_log)  # Reverse log-transform

        # Return the prediction
        return render_template('index.html', prediction_text=f'Predicted Price: {prediction[0][0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)