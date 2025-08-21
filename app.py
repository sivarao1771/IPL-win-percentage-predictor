# Import necessary libraries
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import os # Import the os module for accessing environment variables

# Initialize the Flask application
# Flask is a micro web framework for Python.
app = Flask(__name__)
# CORS (Cross-Origin Resource Sharing) allows your frontend to make requests to this backend server.
CORS(app)

# --- Load the exported machine learning artifacts ---
# This block loads the trained models and preprocessors you created in your notebook.
# It's wrapped in a try-except block to handle errors if files are missing.
try:
    rf_model = joblib.load('rf_model.joblib')
    ann_model = load_model('ann_model.h5')
    preprocessor = joblib.load('preprocessor.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Models and preprocessors loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    rf_model = ann_model = preprocessor = scaler = None

# --- Re-implement the Logistic Regression model from the notebook ---
# These are the learned parameters (weights and bias) from your logistic regression model.
# They are hardcoded here because the model is a simple mathematical formula.
lr_weights = np.array([-0.14, -0.21, -0.04, -0.01, -0.09, 0.04, -0.12, 0.03, -0.07, 0.13, 0.19, 0.03, 0.01, 0.08, 0.02, 0.01, 0.04, 0.03, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0, -1.02, 0.45, 0.98, 0.15, 0.1, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
lr_bias = 0.02
# These are the mean and standard deviation values from your training data, used for scaling.
X_train_mean = np.array([0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 92.2, 59.6, 7.5, 172.5, 8.1, 10.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
X_train_std = np.array([0.3, 0.3, 0.2, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0, 50.5, 33.3, 2.3, 30.5, 2.9, 4.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Sigmoid function converts a number into a probability between 0 and 1.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# This function takes the preprocessed data and calculates the win probability using the LR model.
def predict_logistic_regression(scaled_vector):
    safe_std = np.where(X_train_std == 0, 1, X_train_std)
    manual_scaled_vector = (scaled_vector - X_train_mean) / safe_std
    z = np.dot(manual_scaled_vector, lr_weights) + lr_bias
    return sigmoid(z)

# --- Define API Routes ---

# This route serves the main webpage (index.html).
# The '@app.route' decorator tells Flask what URL should trigger this function.
@app.route('/')
def home():
    # render_template looks for the file in a 'templates' folder.
    return render_template('index.html')

# This route handles the prediction requests from the frontend.
# It only accepts POST requests, which is standard for sending data.
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the models were loaded correctly at startup.
    if not all([rf_model, ann_model, preprocessor, scaler]):
        return jsonify({'error': 'Models not loaded. Server is not ready.'}), 500

    # Get the JSON data sent from the website.
    data = request.get_json(force=True)
    
    # Convert the incoming data into a pandas DataFrame for preprocessing.
    input_df = pd.DataFrame([data], columns=['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'runs_target', 'crr', 'rrr'])

    # Use the loaded preprocessors to transform the new data in the same way as the training data.
    processed_vector = preprocessor.transform(input_df)
    scaled_vector_ann = scaler.transform(processed_vector)

    # --- Get predictions from each of the three models ---
    rf_prob = rf_model.predict_proba(processed_vector)[0][1] # Probability of class 1 (win)
    ann_prob = ann_model.predict(scaled_vector_ann)[0][0]
    lr_prob = predict_logistic_regression(processed_vector[0])

    # --- Calculate the weighted average of the predictions ---
    # Weights are based on the accuracy of each model from your notebook.
    final_prob = (rf_prob * 0.45) + (ann_prob * 0.45) + (lr_prob * 0.10)

    # Prepare the response to send back to the website.
    response = {
        'final_prediction': final_prob,
        'model_breakdown': {
            'random_forest': rf_prob,
            'ann': float(ann_prob), # Convert numpy float to standard float for JSON
            'logistic_regression': lr_prob
        }
    }
    
    # Send the response as a JSON object.
    return jsonify(response)

# --- Run the server ---
# This block ensures the server only runs when the script is executed directly.
if __name__ == '__main__':
    # For deployment, the web server (like Render) will set a PORT environment variable.
    # We use that if it exists, otherwise default to 5000 for local development.
    port = int(os.environ.get('PORT', 5000))
    # 'host=0.0.0.0' makes the server accessible from other devices on your network.
    app.run(host='0.0.0.0', port=port)
