from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("models/cricket_score_model.pkl")

print("MODEL LOADED SUCCESSFULLY")

# Home route
@app.route("/")
def home():
    return "Cricket Score Prediction API Running"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data
    data = request.get_json()

    # Convert input into DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)

    # Return response
    return jsonify({
        "Predicted Score": float(prediction[0])
    })

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)