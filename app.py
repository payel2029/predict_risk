from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("stacking_ensemble_model.pkl")

@app.route('/')
def home():
    return "Maternal Health Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Expecting JSON: {"features": [Age, SystolicBP, Diastolic, BS, BodyTemp, BMI,
    #  PreviousComplications, PreexistingDiabetes, GestationalDiabetes, MentalHealth, HeartRate]}
    
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]

    return jsonify({"prediction": int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
