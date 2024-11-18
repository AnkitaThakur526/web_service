from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from flask_jwt_extended import (
    JWTManager,
    jwt_required,
    create_access_token,
    get_jwt_identity,
)
import numpy as np
import pickle
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

# Initialize the Flask application
app = Flask(__name__)

CORS(app)

# Configuring the app
app.config["MONGO_URI"] = (
    "mongodb+srv://ankita:PdBdpjh8kUwDfZNL@cluster0.mqcwo.mongodb.net/elderlycare?retryWrites=true&w=majority&appName=Cluster0"  # MongoDB URI
)
app.config["SECRET_KEY"] = "Tatakae"  # Secret key for encoding JWT

# Initialize extensions
mongo = PyMongo(app)
jwt = JWTManager(app)

# Load the Naive Bayes model and other necessary files
with open("naive_bayes_model.pkl", "rb") as model_file:
    nb_model = pickle.load(model_file)

with open("label_encoders.pkl", "rb") as enc_file:
    label_enc, exercise_enc = pickle.load(enc_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Condition mapping for display
condition_mapping = {0: "Healthy", 1: "Near Risk", 2: "Obese", 3: "Highly Risk"}


# Route for registering new users
@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    print(data)
    email = data.get("email")
    password = data.get("password")

    if mongo.db.users.find_one({"email": email}):
        return jsonify({"message": "User already exists"}), 400

    hashed_password = generate_password_hash(password)

    mongo.db.users.insert_one({"email": email, "password": hashed_password})

    return jsonify({"message": "Registration successful"}), 201


# Route for logging in users
@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    user = mongo.db.users.find_one({"email": email})

    if not user or not check_password_hash(user["password"], password):
        return jsonify({"message": "Invalid credentials"}), 401

    # Create JWT token
    access_token = create_access_token(
        identity=email, expires_delta=datetime.timedelta(days=1)
    )

    return jsonify({"token": access_token}), 200


# Route to make predictions
@app.route("/api/predict", methods=["POST"])
@jwt_required()
def predict():
    current_user = get_jwt_identity()  # Get current logged-in user

    # Get input data from the request
    data = request.get_json()
    print(data)
    age = int(data["age"])
    sex = data["sex"]
    cp = int(data["cp"])
    trestbps = float(data["trestbps"])
    chol = float(data["chol"])
    fbs = int(data["fbs"])
    restecg = int(data["restecg"])
    thalach = float(data["thalach"])
    exang = int(data["exang"])
    oldpeak = float(data["oldpeak"])
    slope = int(data["slope"])
    ca = int(data["ca"])
    thal = int(data["thal"])

    # Encode the inputs
    sex_encoded = label_enc.transform([sex])[0]

    # Prepare input for prediction
    input_data = np.array(
        [
            [
                age,
                sex_encoded,
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
            ]
        ]
    )

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict using the Naive Bayes model
    prediction_code = nb_model.predict(input_data_scaled)[0]
    prediction_result = condition_mapping[prediction_code]

    # Save prediction in MongoDB
    mongo.db.predictions.insert_one(
        {
            "user": current_user,
            "prediction_result": prediction_result,
            "input_data": data,
        }
    )

    return jsonify({"prediction": prediction_result}), 200


# Route to fetch all predictions for the logged-in user (Dashboard)
@app.route("/api/predictions", methods=["GET"])
@jwt_required()
def get_predictions():
    current_user = get_jwt_identity()  # Get current logged-in user

    predictions = list(mongo.db.predictions.find({"user": current_user}, {"_id": 0}))

    return jsonify({"predictions": predictions}), 200


# Main entry point
if __name__ == "__main__":
    app.run(host="127.0.0.1",port=5000,debug=True)
    
