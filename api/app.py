from flask import Flask, request, jsonify,render_template
from utils.feature_pipeline import validate_input, prepare_features
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
app = Flask(
    __name__,
    template_folder="../templates"
)


# Load trained model
model = joblib.load("models/cricket_score_model.pkl")

logging.info("MODEL LOADED SUCCESSFULLY")

# Home route

@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.form.to_dict()

        for key in data:
            data[key] = float(data[key])

        is_valid, missing = validate_input(data)

        if not is_valid:

            return jsonify({
                "error": f"Missing features: {missing}"
            }), 400

        features_df = prepare_features(data)

        prediction = model.predict(features_df)

        return f"Predicted Score: {prediction[0]}"

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500





# Run app
if __name__ == "__main__":
    app.run(debug=True)