from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("models/cricket_score_model.pkl")

logging.info("MODEL LOADED SUCCESSFULLY")

# Home route
@app.route("/")
def home():
    return "Cricket Score Prediction API Running"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Receive JSON input
        data = request.get_json()

        logging.info(f"Incoming data: {data}")

        # Validation
       	required_features = ["Match ID","Overs Played","Wickets Lost","Run Rate","Home/Away",    "Opponent Strength",    "Pitch Condition", "Weather"]

        for feature in required_features:
            if feature not in data:
                logging.error(f"Missing feature: {feature}")

                return jsonify({
                    "error": f"Missing feature: {feature}"
                })

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(input_data)

        logging.info(f"Prediction successful: {prediction[0]}")

        # Return prediction
        return jsonify({
            "Predicted Score": float(prediction[0])
        })

    except Exception as e:

        logging.error(f"Error occurred: {str(e)}")

        return jsonify({
            "error": str(e)
        })

# Run app
if __name__ == "__main__":
    app.run(debug=True)